"""
ped_crosswalk_tag.py
Sistema de faixa de pedestres com:
- YOLOv8 (apenas pessoas) e ROI desenhável da faixa
- Simulador de TAG (teclado 't'/'1'..'9', botão na tela, arquivo tag.trigger)
- FSM:
    VEH_GREEN → VEH_YEL → ALL_RED_TO_PED → PED_GREEN → ALL_RED_TO_VEH → VEH_GREEN
- TAG apenas acelera o fim do VEH_GREEN (pré-empção)
- PED_GREEN: bônus inicial se já houver pessoa e EXTENSÃO contínua enquanto houver pessoa
- HUD mostra estados de CARROS, PEDESTRE e “Pessoa na faixa: SIM/NAO”
- Modo --fast: descarta frames RTSP e faz inferência em resolução reduzida

Instalação:
    pip install ultralytics opencv-python numpy

Como rodar:
    python ped_crosswalk_tag.py
    python ped_crosswalk_tag.py --fast
    python ped_crosswalk_tag.py reset   # apaga ROI e redesenha

Controles:
    q       -> sair
    t       -> simula TAG
    1..9    -> simula TAG com ID
    Clique no botão "TAG" na tela
    touch tag.trigger -> dispara uma vez (crie o arquivo vazio na pasta)
"""

import sys, os, time, json
from enum import Enum
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ===================== CONFIG BÁSICA =====================
YOLO_MODEL_PATH = "yolo11s.pt"

# Fonte de vídeo (RTSP da câmera) OU Webcam:
#CAMERA_URL = "rtsp://admin:QT9RJ462@192.168.0.108:554/cam/realmonitor?channel=1&subtype=1"
CAMERA_URL = 1  # descomente para usar webcam

ROI_JSON = "roi_pedestre.json"

# Classes (COCO): 0=person
CL_PESSOA = {0}
DET_CONF = 0.45  # levemente menor para capturar pedestres mais difíceis

# Cores
C_PESSOA = (0, 255, 0)
C_ROI    = (255, 0, 255)

MAIN_WIN = "Faixa Pedestre + TAG"

# ===================== Simulador de TAG =====================
class TagSimulator:
    TAG_BUTTON_RECT = (20, 120, 180, 170)
    FILE_TRIGGER = "tag.trigger"
    COOLDOWN = 0.8

    def __init__(self):
        self._pulse = False
        self._last  = 0.0
        self._tagid = None

    def _ok(self): 
        return (time.time() - self._last) >= self.COOLDOWN

    def keypress(self, k):
        if not self._ok(): 
            return
        if k == ord('t'):
            self._pulse, self._tagid, self._last = True, None, time.time()
        if ord('1') <= k <= ord('9'):
            self._pulse, self._tagid, self._last = True, int(chr(k)), time.time()

    def click(self, x, y):
        if not self._ok(): 
            return
        x1,y1,x2,y2 = self.TAG_BUTTON_RECT
        if x1<=x<=x2 and y1<=y<=y2:
            self._pulse, self._tagid, self._last = True, None, time.time()

    def file_poll(self):
        if not self._ok(): 
            return
        if os.path.exists(self.FILE_TRIGGER):
            try: 
                os.remove(self.FILE_TRIGGER)
            except: 
                pass
            self._pulse, self._tagid, self._last = True, None, time.time()

    def poll(self):
        if self._pulse:
            self._pulse = False
            return True, self._tagid
        return False, None

    def draw_button(self, frame):
        x1,y1,x2,y2 = self.TAG_BUTTON_RECT
        # desenha somente a área do botão para economizar CPU
        overlay = frame.copy()
        cv2.rectangle(overlay,(x1,y1),(x2,y2),(60,200,60),-1)
        roi = frame[y1:y2, x1:x2]
        ov_roi = overlay[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = cv2.addWeighted(ov_roi,0.4,roi,0.6,0)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(80,255,80),2)
        cv2.putText(frame,"TAG",(x1+12,y1+35),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2)
        cv2.putText(frame,"[T] ou clique",(x1+10,y2-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(230,230,230),1)

# ===================== FSM do Semáforo =====================
class State(Enum):
    VEH_GREEN=1
    VEH_YEL=2
    ALL_RED_TO_PED=3
    PED_GREEN=4
    ALL_RED_TO_VEH=5

# Tempos (s)
VEH_GREEN_DEFAULT=20       # verde carros normal
VEH_YEL_TIME=5             # amarelo carros
ALL_RED_CLEAR=3            # intertravamentos (antes e depois do pedestre)
PED_GREEN_BASE=12          # pedestre base
PED_EXT_STEP=5             # extensão por detecção contínua
PEDESTRIAN_CLEAR_GRACE=5   # faixa livre por Xs para encerrar pedestre
PED_ENTRY_BOOST=5          # bônus inicial se já houver pessoa ao abrir pedestre

# Preempção por TAG (acelera fechamento do verde dos carros)
MIN_GREEN_BEFORE_PREEMPT=5
PREEMPT_CUT_TO=5

class TrafficLightFSM:
    """
    Ciclo: VEH_GREEN → VEH_YEL → ALL_RED_TO_PED → PED_GREEN → ALL_RED_TO_VEH → VEH_GREEN
    - TAG em VEH_GREEN só antecipa a ida ao VEH_YEL (preempção).
    - Em PED_GREEN, o tempo é estendido enquanto houver pessoa NA FAIXA.
    """
    def __init__(self):
        self.state=State.VEH_GREEN
        self.t0=time.monotonic(); self.tgt=VEH_GREEN_DEFAULT
        self.tag_pending=False
        self.last_seen_person_ts=None
        # Saídas simuladas
        self.outputs=dict(veh_g=True,veh_y=False,veh_r=False,ped_w=False,ped_dw=True)

    def _reset(self,dur): 
        self.t0=time.monotonic(); self.tgt=dur
    def _elapsed(self): 
        return time.monotonic()-self.t0
    def remaining(self): 
        return max(0.0,self.tgt-self._elapsed())
    def request_tag(self): 
        self.tag_pending=True

    def _outs(self,vg,vy,vr,pw,pdw):
        self.outputs=dict(veh_g=vg,veh_y=vy,veh_r=vr,ped_w=pw,ped_dw=pdw)

    def update(self,pessoa_na_faixa:bool):
        now=time.monotonic()

        # TAG durante VERDE carros => corta p/ amarelo (sem consumir tag aqui)
        if self.state==State.VEH_GREEN and self.tag_pending:
            if self._elapsed()>=MIN_GREEN_BEFORE_PREEMPT:
                self.state=State.VEH_YEL
                self._reset(PREEMPT_CUT_TO)

        # VEHICLE GREEN
        if self.state==State.VEH_GREEN:
            self._outs(True,False,False,False,True)
            if self._elapsed()>=self.tgt:
                self.state=State.VEH_YEL; self._reset(VEH_YEL_TIME)

        # VEHICLE YELLOW
        elif self.state==State.VEH_YEL:
            self._outs(False,True,False,False,True)
            if self._elapsed()>=self.tgt:
                self.state=State.ALL_RED_TO_PED; self._reset(ALL_RED_CLEAR)

        # ALL RED → preparar para abrir pedestre (pedestre AINDA FECHADO AQUI)
        elif self.state==State.ALL_RED_TO_PED:
            self._outs(False,False,True,False,True)  # all-red total
            if self._elapsed()>=self.tgt:
                self.state=State.PED_GREEN
                # bônus se já houver pessoa agora
                extra = PED_ENTRY_BOOST if pessoa_na_faixa else 0
                self._reset(PED_GREEN_BASE + extra)
                # inicia/zera marcador de presença
                self.last_seen_person_ts = now if pessoa_na_faixa else None
                self.tag_pending=False  # consome tag (se veio)

        # PED GREEN (estende enquanto houver pessoa)
        elif self.state==State.PED_GREEN:
            self._outs(False,False,True,True,False)
            if pessoa_na_faixa:
                self.last_seen_person_ts = now
            base_done = self._elapsed() >= self.tgt
            nobody_recent = (self.last_seen_person_ts is None) or ((now - self.last_seen_person_ts) >= PEDESTRIAN_CLEAR_GRACE)

            if base_done:
                if nobody_recent:
                    self.state=State.ALL_RED_TO_VEH; self._reset(ALL_RED_CLEAR)
                else:
                    # Estende adicionando segundos ao alvo atual (em vez de resetar com remaining())
                    self.tgt += PED_EXT_STEP

        # ALL RED → voltar carros
        elif self.state==State.ALL_RED_TO_VEH:
            self._outs(False,False,True,False,True)
            if self._elapsed()>=self.tgt:
                self.state=State.VEH_GREEN
                self._reset(VEH_GREEN_DEFAULT)
                self.last_seen_person_ts=None
        return self.state

    # HUD à direita (inclui “Pessoa na faixa”)
    @staticmethod
    def _put_right(frame,text,y,scale=0.9,color=(255,255,255),th=2):
        (w,_),_ = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,th)
        x = frame.shape[1]-20-w
        cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),th+1)
        cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,th)

    def overlay(self,frame,pessoa_na_faixa:bool):
        car_state = "VERDE" if self.outputs["veh_g"] else ("AMARELO" if self.outputs["veh_y"] else "VERMELHO")
        ped_state = "CAMINHE" if self.outputs["ped_w"] else "ESPERE"
        state_name={
            State.VEH_GREEN:"CARROS: VERDE",
            State.VEH_YEL:"CARROS: AMARELO",
            State.ALL_RED_TO_PED:"TRANSIÇÃO → PEDESTRE",
            State.PED_GREEN:"PEDESTRE: ABERTO",
            State.ALL_RED_TO_VEH:"TRANSIÇÃO → CARROS",
        }[self.state]
        self._put_right(frame,f"{state_name} | T-{int(round(self.remaining()))}s",40)
        self._put_right(frame,f"Carros: {car_state}",70,0.85,(200,255,200))
        self._put_right(frame,f"Pedestre: {ped_state}",100,0.85,(200,220,255))
        self._put_right(frame,f"Pessoa na faixa: {'SIM' if pessoa_na_faixa else 'NAO'}",130,0.85,(255,255,255))

# ===================== ROI e utilitários =====================
_clicked: List[Tuple[int,int]]=[]; _selecting=False
tag_sim: TagSimulator|None=None

def save_roi(pts,path): 
    with open(path,"w") as f:
        f.write(json.dumps({"points":pts}))

def load_roi(path): 
    with open(path,"r") as f:
        data=json.load(f)
    return [tuple(p) for p in data["points"]]

def in_poly(pt,poly): 
    return cv2.pointPolygonTest(np.array(poly,np.int32),pt,False)>=0

def draw_poly(frame,poly,color,fill=True):
    pts=np.array(poly,np.int32)
    if fill:
        ov=frame.copy(); 
        cv2.fillPoly(ov,[pts],color)
        frame[:]=cv2.addWeighted(ov,0.2,frame,0.8,0)
    cv2.polylines(frame,[pts],True,color,2)

def mouse_cb(ev,x,y,flags,param):
    global _clicked,_selecting, tag_sim
    if param=="roi":
        if ev==cv2.EVENT_LBUTTONDOWN: 
            _clicked.append((x,y)); _selecting=True
        elif ev==cv2.EVENT_RBUTTONDOWN and _clicked: 
            _clicked.pop()
        elif ev==cv2.EVENT_MBUTTONDOWN: 
            _selecting=False
    else:
        if ev==cv2.EVENT_LBUTTONDOWN and tag_sim is not None:
            tag_sim.click(x,y)

# ===================== MAIN =====================
def main():
    global tag_sim

    FAST = ("--fast" in [a.lower() for a in sys.argv])
    if len(sys.argv)>1 and sys.argv[1].lower()=="reset":
        if os.path.exists(ROI_JSON): 
            os.remove(ROI_JSON); 
            print("ROI apagada.")

    model=YOLO(YOLO_MODEL_PATH)
    # Pequenas otimizações (se suportado)
    try: 
        model.fuse()
    except: 
        pass

    cap=cv2.VideoCapture(CAMERA_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)


    if not cap.isOpened(): 
        print("Nao abriu camera"); 
        return

    tag_sim=TagSimulator()
    fsm=TrafficLightFSM()

    # ROI
    if os.path.exists(ROI_JSON):
        try:
            roi=load_roi(ROI_JSON); 
            print("ROI:",roi)
        except Exception as e:
            print("Falha ao ler ROI, redesenhando...", e)
            roi=None
    else:
        roi=None

    if roi is None:
        cv2.namedWindow("Selecione ROI"); 
        cv2.setMouseCallback("Selecione ROI",mouse_cb,param="roi")
        while True:
            # descarta frames pendentes (ajuda em RTSP)
            for _ in range(2): cap.grab()
            ok,frame=cap.read()
            if not ok: break
            prev=frame.copy()
            for p in _clicked: 
                cv2.circle(prev,p,4,(0,255,255),-1)
            if len(_clicked)>=2:
                cv2.polylines(prev,[np.array(_clicked,np.int32)],False,(0,255,255),2)
            cv2.putText(prev,"ESQ:add | DIR:desfaz | MEIO:finalizar",(20,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(50,255,255),2)
            cv2.imshow("Selecione ROI",prev)
            key=cv2.waitKey(1)&0xFF
            if not _selecting and len(_clicked)>=3:
                roi=_clicked.copy(); 
                save_roi(roi,ROI_JSON); 
                cv2.destroyWindow("Selecione ROI"); 
                break
            if key==ord('q') or key==27: 
                return
    if roi is None: 
        return

    cv2.namedWindow(MAIN_WIN); 
    cv2.setMouseCallback(MAIN_WIN,mouse_cb,param=None)

    # Parâmetros de inferência
    infer_scale = 0.75 if FAST else 1.0
    imgsz = 640

    while True:
        # === leitura de frame ===
        if FAST:
            # descarta frames velhos (reduz latência em RTSP)
            for _ in range(2): 
                cap.grab()
        ok,frame=cap.read()
        if not ok: 
            break

        draw_poly(frame,roi,C_ROI,fill=True)

        # === detecção apenas de pessoas ===
        if infer_scale != 1.0:
            infer_frame = cv2.resize(frame, None, fx=infer_scale, fy=infer_scale, interpolation=cv2.INTER_LINEAR)
        else:
            infer_frame = frame

        res = model(infer_frame, conf=DET_CONF, imgsz=imgsz, verbose=False)[0]
        pessoa_na_faixa=False

        if res.boxes is not None and len(res.boxes) > 0:
            # mapeia coords de volta para o frame original, se necessário
            sx = (frame.shape[1] / infer_frame.shape[1])
            sy = (frame.shape[0] / infer_frame.shape[0])

            for b in res.boxes:
                cls = int(b.cls[0])
                if cls not in CL_PESSOA:
                    continue
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                if infer_scale != 1.0:
                    x1 = int(x1 * sx); x2 = int(x2 * sx)
                    y1 = int(y1 * sy); y2 = int(y2 * sy)
                cx,cy=(x1+x2)//2,(y1+y2)//2
                inside=in_poly((cx,cy),roi)

                # desenho básico para depuração
                color = C_PESSOA if inside else (120,220,120)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.circle(frame,(cx,cy),4,color,-1)

                if inside: 
                    pessoa_na_faixa=True

        # === TAG simulação ===
        tag_sim.file_poll()

        # === FSM e HUD ===
        fsm.update(pessoa_na_faixa)
        fsm.overlay(frame, pessoa_na_faixa)
        tag_sim.draw_button(frame)

        # === janela e teclado (ordem correta: imshow -> waitKey) ===
        cv2.imshow(MAIN_WIN,frame)
        key=cv2.waitKey(1)&0xFF
        if key==ord('q') or key==27: 
            break
        tag_sim.keypress(key)
        pulse,_=tag_sim.poll()
        if pulse: 
            fsm.request_tag()   # acelera fechamento do verde carros se aplicável

    cap.release(); 
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
