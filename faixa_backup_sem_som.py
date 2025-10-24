"""
ped_crosswalk_tag.py
Faixa de pedestres com:
- YOLOv8 (pessoas + veículos) e ROI da faixa
- Simulador de TAG (teclado 't'/'1'..'9', botão na tela, arquivo tag.trigger)
- FSM:
    VEH_GREEN → VEH_YEL → ALL_RED_TO_PED → PED_GREEN → ALL_RED_TO_VEH → VEH_GREEN
- TAG apenas acelera o fim do VEH_GREEN
- PED_GREEN: bônus inicial se já houver pessoa e EXTENSÃO contínua enquanto houver pessoa
- HUD mostra estados de CARROS, PEDESTRE e “Pessoa na faixa: SIM/NAO”

Instalação:
    pip install ultralytics opencv-python numpy

Como rodar:
    python ped_crosswalk_tag.py
    python ped_crosswalk_tag.py reset   # apaga ROI e redesenha

Controles:
    q       -> sair
    t       -> simula TAG
    1..9    -> simula TAG com ID
    Clique no botão "TAG" na tela
    touch/tag.trigger -> dispara uma vez
"""

import sys, os, time, json
from enum import Enum
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ===================== CONFIG BÁSICA =====================
YOLO_MODEL_PATH = "yolov8n.pt"

# Fonte de vídeo (RTSP da IM4C) OU Webcam:
CAMERA_URL = "rtsp://admin:QT9RJ462@192.168.0.108:554/cam/realmonitor?channel=1&subtype=1"
#CAMERA_URL = 0  # descomente para usar webcam

ROI_JSON = "roi_pedestre.json"

# Classes (COCO): 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
CL_PESSOA = {0}
CL_VEICULO = {2, 3, 5, 7}
DET_CONF = 0.5

# Cores
C_PESSOA = (0,255,0)
C_VEICULO= (255,255,0)
C_EMERG  = (0,0,255)
C_ROI    = (255,0,255)

# Faixas de cor (HSV) para "vermelho" (emergência visual)
RED_RANGES = [
    (np.array([0, 80, 80]),   np.array([10, 255, 255])),
    (np.array([160, 80, 80]), np.array([179, 255, 255])),
]
RED_RATIO_THR = 0.12

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

    def _ok(self): return (time.time() - self._last) >= self.COOLDOWN

    def keypress(self, k):
        if not self._ok(): return
        if k == ord('t'):
            self._pulse, self._tagid, self._last = True, None, time.time()
        if ord('1') <= k <= ord('9'):
            self._pulse, self._tagid, self._last = True, int(chr(k)), time.time()

    def click(self, x, y):
        if not self._ok(): return
        x1,y1,x2,y2 = self.TAG_BUTTON_RECT
        if x1<=x<=x2 and y1<=y<=y2:
            self._pulse, self._tagid, self._last = True, None, time.time()

    def file_poll(self):
        if not self._ok(): return
        if os.path.exists(self.FILE_TRIGGER):
            try: os.remove(self.FILE_TRIGGER)
            except: pass
            self._pulse, self._tagid, self._last = True, None, time.time()

    def poll(self):
        if self._pulse:
            self._pulse = False
            return True, self._tagid
        return False, None

    def draw_button(self, frame):
        x1,y1,x2,y2 = self.TAG_BUTTON_RECT
        overlay = frame.copy()
        cv2.rectangle(overlay,(x1,y1),(x2,y2),(60,200,60),-1)
        frame[:] = cv2.addWeighted(overlay,0.4,frame,0.6,0)
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

    def _reset(self,dur): self.t0=time.monotonic(); self.tgt=dur
    def _elapsed(self): return time.monotonic()-self.t0
    def remaining(self): return max(0.0,self.tgt-self._elapsed())
    def request_tag(self): self.tag_pending=True

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

        # ALL RED → abrir pedestre
        elif self.state==State.ALL_RED_TO_PED:
            self._outs(False,False,True,True,False)  # carros vermelhos, pedestre já verde
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

def save_roi(pts,path): open(path,"w").write(json.dumps({"points":pts}))
def load_roi(path): return [tuple(p) for p in json.load(open(path))["points"]]
def in_poly(pt,poly): return cv2.pointPolygonTest(np.array(poly,np.int32),pt,False)>=0

def draw_poly(frame,poly,color,fill=True):
    pts=np.array(poly,np.int32)
    if fill:
        ov=frame.copy(); cv2.fillPoly(ov,[pts],color); frame[:]=cv2.addWeighted(ov,0.2,frame,0.8,0)
    cv2.polylines(frame,[pts],True,color,2)

def red_ratio(frame_bgr, x1,y1,x2,y2):
    h,w=frame_bgr.shape[:2]
    x1=max(0,x1); y1=max(0,y1); x2=min(w-1,x2); y2=min(h-1,y2)
    if x2<=x1 or y2<=y1: return 0.0
    crop = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    masks=[cv2.inRange(hsv,lo,hi) for lo,hi in RED_RANGES]
    m=masks[0]
    for k in masks[1:]: m=cv2.bitwise_or(m,k)
    return float(np.count_nonzero(m))/float(m.size)

def mouse_cb(ev,x,y,flags,param):
    global _clicked,_selecting, tag_sim
    if param=="roi":
        if ev==cv2.EVENT_LBUTTONDOWN: _clicked.append((x,y)); _selecting=True
        elif ev==cv2.EVENT_RBUTTONDOWN and _clicked: _clicked.pop()
        elif ev==cv2.EVENT_MBUTTONDOWN: _selecting=False
    else:
        if ev==cv2.EVENT_LBUTTONDOWN and tag_sim is not None:
            tag_sim.click(x,y)

# ===================== MAIN =====================
def main():
    global tag_sim
    if len(sys.argv)>1 and sys.argv[1].lower()=="reset":
        if os.path.exists(ROI_JSON): os.remove(ROI_JSON); print("ROI apagada.")

    model=YOLO(YOLO_MODEL_PATH)
    cap=cv2.VideoCapture(CAMERA_URL) if not isinstance(CAMERA_URL,tuple) else cv2.VideoCapture(*CAMERA_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    if not cap.isOpened(): print("Nao abriu camera"); return

    tag_sim=TagSimulator()
    fsm=TrafficLightFSM()

    # ROI
    if os.path.exists(ROI_JSON):
        roi=load_roi(ROI_JSON); print("ROI:",roi)
    else:
        cv2.namedWindow("Selecione ROI"); cv2.setMouseCallback("Selecione ROI",mouse_cb,param="roi")
        roi=None
        while True:
            for _ in range(3): cap.grab()
            ok,frame=cap.read()
            if not ok: break
            prev=frame.copy()
            for p in _clicked: cv2.circle(prev,p,4,(0,255,255),-1)
            if len(_clicked)>=2:
                cv2.polylines(prev,[np.array(_clicked,np.int32)],False,(0,255,255),2)
            cv2.putText(prev,"ESQ:add | DIR:desfaz | MEIO:finalizar",(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(50,255,255),2)
            cv2.imshow("Selecione ROI",prev)
            if not _selecting and len(_clicked)>=3:
                roi=_clicked.copy(); save_roi(roi,ROI_JSON); cv2.destroyWindow("Selecione ROI"); break
            if cv2.waitKey(1)&0xFF==ord('q'): return
    if roi is None: return

    cv2.namedWindow(MAIN_WIN); cv2.setMouseCallback(MAIN_WIN,mouse_cb,param=None)

    while True:
        ok,frame=cap.read()
        if not ok: break

        draw_poly(frame,roi,C_ROI,fill=True)

        # -------- Detecção (pessoa na ROI + veículos p/ HUD)
        res=model(frame, conf=DET_CONF)[0]
        pessoa_na_faixa=False

        if res.boxes is not None:
            names=res.names if hasattr(res,"names") else {}
            for b in res.boxes:
                cls=int(b.cls[0]); conf=float(b.conf[0])
                x1,y1,x2,y2=map(int,b.xyxy[0]); cx,cy=(x1+x2)//2,(y1+y2)//2
                label=names.get(cls,str(cls))
                inside=in_poly((cx,cy),roi)

                if cls in CL_PESSOA:
                    color = C_PESSOA if inside else (120,220,120)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    cv2.circle(frame,(cx,cy),4,color,-1)
                    cv2.putText(frame,f"{label} {conf:.2f}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                    if inside: pessoa_na_faixa=True
                elif cls in CL_VEICULO:
                    rr = red_ratio(frame,x1,y1,x2,y2)
                    is_red = rr>=RED_RATIO_THR
                    is_emergency = is_red and (cls==2 or cls==7)   # carro/caminhao vermelhos (apenas visual)
                    color = C_EMERG if is_emergency else C_VEICULO
                    tagtxt=" | EMERGENCY" if is_emergency else ""
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    cv2.circle(frame,(cx,cy),4,color,-1)
                    cv2.putText(frame,f"{label} {conf:.2f}{tagtxt}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        # -------- TAG simulação
        tag_sim.file_poll()
        key=cv2.waitKey(1)&0xFF
        tag_sim.keypress(key)
        pulse,_=tag_sim.poll()
        if pulse: fsm.request_tag()   # acelera fechamento do verde carros se aplicável

        # -------- FSM e HUD
        fsm.update(pessoa_na_faixa)
        fsm.overlay(frame, pessoa_na_faixa)
        tag_sim.draw_button(frame)

        cv2.imshow(MAIN_WIN,frame)
        if key==ord('q') or key==27: break

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
