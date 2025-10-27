"""
ped_crosswalk_tag.py
Sistema de faixa de pedestres com:
- YOLOv8 (apenas pessoas) e ROI desenhável da faixa
- Integração por CABO com ESP32 (RFID RC522 + BOTÃO físico):
    * PC → ESP: OUT,<b0..b4>  (veh_g,veh_y,veh_r,ped_w,ped_dw)
    * PC → ESP: PING          (ESP responde PONG)
    * ESP → PC: TAG,<uidhex>  (cartão lido)
    * ESP → PC: BTN,PED       (botão físico pressionado)
    * ESP → PC: HB            (heartbeat 1s)
- Simulador de TAG local (teclado 't'/'1'..'9', botão na tela, arquivo tag.trigger)
- FSM:
    VEH_GREEN → VEH_YEL → ALL_RED_TO_PED → PED_GREEN → ALL_RED_TO_VEH → VEH_GREEN
- Pré-empção (TAG/BOTÃO): reduz o VERDE dos carros para 5s restantes (não vai direto ao amarelo)
- PED_GREEN: bônus inicial se já houver pessoa e EXTENSÃO contínua enquanto houver pessoa
- HUD: estados, “Pessoa na faixa: SIM/NAO”, mini semáforos, indicadores de TAG/BOTÃO e UID
- CLI webcam: --480p / --720p / --1080p / --width/--height/--fps
- Modo --fast: descarta frames antigos e faz inferência em frame reduzido

Instalação:
    pip install ultralytics opencv-python numpy pyserial

Como rodar:
    python ped_crosswalk_tag.py
    python ped_crosswalk_tag.py --fast
    python ped_crosswalk_tag.py --720p
    python ped_crosswalk_tag.py reset   # apaga ROI e redesenha

Controles:
    q       -> sair
    t       -> simula TAG
    1..9    -> simula TAG com ID
    Clique no botão "TAG" na tela
    touch tag.trigger -> dispara uma vez (crie o arquivo vazio na pasta)
"""

import sys, os, time, json, argparse
from enum import Enum
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Serial / threading
import serial, serial.tools.list_ports
import threading, queue

# ===================== CONFIG BÁSICA =====================
YOLO_MODEL_PATH = "yolo11n.pt"
CAMERA_URL = 1  # use 0 / 1 / 2... para webcams
ROI_JSON = "roi_pedestre.json"
MANUAL_RES = 1080   # escolha: 480 / 720 / 1080 / ou None para ignorar

# Classes (COCO): 0=person
CL_PESSOA = {0}
DET_CONF = 0.45  # levemente menor para capturar pedestres mais difíceis

# Cores
C_PESSOA = (0, 255, 0)
C_ROI    = (255, 0, 255)

MAIN_WIN = "Faixa Pedestre + TAG"

# ===================== Simulador de TAG (local) =====================
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
VEH_YEL_TIME=5             # amarelo carros (valor base — será substituído pela janela de pisca)
ALL_RED_CLEAR=3            # intertravamentos (antes e depois do pedestre)
PED_GREEN_BASE=12          # pedestre base
PED_EXT_STEP=5             # extensão por detecção contínua
PEDESTRIAN_CLEAR_GRACE=5   # faixa livre por Xs para encerrar pedestre
PED_ENTRY_BOOST=5          # bônus inicial se já houver pessoa ao abrir pedestre

# Preempção por TAG/BOTÃO
MIN_GREEN_BEFORE_PREEMPT=5   # verde mínimo antes de aceitar pré-empção
PREEMPT_CUT_TO=5             # (não usado diretamente; mantido p/ compatibilidade)

# --- Pisca ---
BLINK_PERIOD_S = 1.0     # 1 segundo total (0.5 on/0.5 off)
BLINK_COUNT = 5          # 5 piscadas (~5s)

class TrafficLightFSM:
    """
    Ciclo: VEH_GREEN → VEH_YEL → ALL_RED_TO_PED → PED_GREEN → ALL_RED_TO_VEH → VEH_GREEN
    - Pré-empção em VERDE carros: reduz o VERDE para restar 5s (não vai direto ao amarelo).
    - VEH_YEL: pisca amarelo por ~5s (5 toques de 1s) antes do vermelho.
    - PED_GREEN: extensão enquanto houver pessoa; ao fechar, pisca verde por ~5s antes de fechar.
    """
    def __init__(self):
        self.state=State.VEH_GREEN
        self.t0=time.monotonic(); self.tgt=VEH_GREEN_DEFAULT
        self.tag_pending=False
        self.last_seen_person_ts=None
        # Saídas
        self.outputs=dict(veh_g=True,veh_y=False,veh_r=False,ped_w=False,ped_dw=True)
        # estados internos de pisca
        self._yel_blink_start = None      # para VEH_YEL
        self._ped_blink_start = None      # para pré-fechamento do PED_GREEN
        self._ped_blinking = False

    def _reset(self,dur): 
        self.t0=time.monotonic(); self.tgt=dur
    def _elapsed(self): 
        return time.monotonic()-self.t0
    def remaining(self): 
        return max(0.0,self.tgt-self._elapsed())
    def request_tag(self): 
        self.tag_pending=True

    # --- Controle de pisca ---
    def _blink_on(self, start_ts, period=BLINK_PERIOD_S):
        if start_ts is None:
            return True
        phase = (time.monotonic() - start_ts) % period
        return phase < (period / 2.0)

    def _outs(self,vg,vy,vr,pw,pdw):
        self.outputs=dict(veh_g=vg,veh_y=vy,veh_r=vr,ped_w=pw,ped_dw=pdw)

    def update(self,pessoa_na_faixa:bool):
        now=time.monotonic()

        # Pré-empção durante VERDE carros: reduzir para 5s restantes (sem trocar de estado)
        if self.state == State.VEH_GREEN and self.tag_pending:
            if self._elapsed() >= MIN_GREEN_BEFORE_PREEMPT:
                self.t0 = time.monotonic()
                self.tgt = 5.0
                self.tag_pending = False  # consumido

        # VEHICLE GREEN
        if self.state==State.VEH_GREEN:
            self._outs(True,False,False,False,True)
            # limpesa de flags de pisca por segurança
            self._yel_blink_start = None
            if self._elapsed()>=self.tgt:
                self.state=State.VEH_YEL
                # ao entrar no amarelo, iniciaremos janela de pisca no bloco do estado

        # VEHICLE YELLOW (pisca ~5s)
        elif self.state == State.VEH_YEL:
            if self._yel_blink_start is None:
                self._yel_blink_start = time.monotonic()
                # garante ~5s de janela de pisca
                self._reset(BLINK_COUNT * BLINK_PERIOD_S)

            yellow_on = self._blink_on(self._yel_blink_start, BLINK_PERIOD_S)
            self._outs(False, yellow_on, False, False, True)

            if self._elapsed() >= self.tgt:
                # fim da janela de pisca → vermelho (ALL_RED_TO_PED)
                self._yel_blink_start = None
                self.state = State.ALL_RED_TO_PED
                self._reset(ALL_RED_CLEAR)

        # ALL RED → preparar para abrir pedestre (pedestre FECHADO aqui)
        elif self.state==State.ALL_RED_TO_PED:
            self._outs(False,False,True,False,True)
            if self._elapsed()>=self.tgt:
                self.state=State.PED_GREEN
                extra = PED_ENTRY_BOOST if pessoa_na_faixa else 0
                self._reset(PED_GREEN_BASE + extra)
                self.last_seen_person_ts = now if pessoa_na_faixa else None
                self.tag_pending=False  # consome tag/botão
                # limpa pisca pedestre
                self._ped_blinking = False
                self._ped_blink_start = None

        # PED GREEN (abre; ao encerrar, pisca verde por ~5s)
        elif self.state==State.PED_GREEN:
            if pessoa_na_faixa:
                self.last_seen_person_ts = now

            base_done = self._elapsed() >= self.tgt
            nobody_recent = (self.last_seen_person_ts is None) or ((now - self.last_seen_person_ts) >= PEDESTRIAN_CLEAR_GRACE)

            if not self._ped_blinking:
                # fase contínua (verde fixo)
                self._outs(False,False,True,True,False)
                if base_done:
                    if nobody_recent:
                        # inicia janela de pisca do pedestre (~5s)
                        self._ped_blinking = True
                        self._ped_blink_start = time.monotonic()
                        self._reset(BLINK_COUNT * BLINK_PERIOD_S)
                    else:
                        # extensão contínua
                        self.tgt += PED_EXT_STEP
            else:
                # fase de pisca do pedestre
                ped_green_on = self._blink_on(self._ped_blink_start, BLINK_PERIOD_S)
                self._outs(False,False,True,ped_green_on,False)

                if self._elapsed() >= self.tgt:
                    # encerra pisca → fecha pedestre
                    self._ped_blinking = False
                    self._ped_blink_start = None
                    self.state=State.ALL_RED_TO_VEH
                    self._reset(ALL_RED_CLEAR)

        # ALL RED → voltar carros
        elif self.state==State.ALL_RED_TO_VEH:
            self._outs(False,False,True,False,True)
            if self._elapsed()>=self.tgt:
                self.state=State.VEH_GREEN
                self._reset(VEH_GREEN_DEFAULT)
                self.last_seen_person_ts=None
                # limpa qualquer pisca remanescente
                self._yel_blink_start = None
                self._ped_blinking = False
                self._ped_blink_start = None

        return self.state

    # HUD helpers
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
            State.VEH_YEL:"CARROS: AMARELO (pisca)",
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

# ---------------- HUD extra: mini semáforos e indicadores ----------------
def draw_light_bulb(img, center, radius, on, color_on, color_off=(60,60,60), thick=-1):
    color = color_on if on else color_off
    cv2.circle(img, center, radius, (0,0,0), 2)
    cv2.circle(img, center, radius-2, color, thick)

def draw_mini_traffic_lights(frame, outs: dict, origin_xy=(20, 20)):
    x, y = origin_xy
    # Caixa carros (vertical: R, Y, G)
    box_w, box_h = 60, 160
    cv2.rectangle(frame, (x, y), (x+box_w, y+box_h), (40,40,40), -1)
    cv2.rectangle(frame, (x, y), (x+box_w, y+box_h), (200,200,200), 2)
    # Lâmpadas carros
    cx = x + box_w//2
    r = 16
    draw_light_bulb(frame, (cx, y+30), r, outs["veh_r"], (0,0,255))     # vermelho
    draw_light_bulb(frame, (cx, y+80), r, outs["veh_y"], (0,255,255))   # amarelo
    draw_light_bulb(frame, (cx, y+130), r, outs["veh_g"], (0,255,0))    # verde
    cv2.putText(frame, "CARROS", (x-2, y+box_h+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,230,230), 1)

    # Caixa pedestre (lado)
    x2 = x + box_w + 20
    box2_w, box2_h = 60, 100
    cv2.rectangle(frame, (x2, y), (x2+box2_w, y+box2_h), (40,40,40), -1)
    cv2.rectangle(frame, (x2, y), (x2+box2_w, y+box2_h), (200,200,200), 2)
    # Lâmpadas pedestre: DON'T WALK (vermelho) em cima, WALK (verde) em baixo
    cx2 = x2 + box2_w//2
    draw_light_bulb(frame, (cx2, y+30), 16, outs["ped_dw"], (0,0,255))
    draw_light_bulb(frame, (cx2, y+70), 16, outs["ped_w"], (0,255,0))
    cv2.putText(frame, "PEDESTRE", (x2-6, y+box2_h+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,230,230), 1)

def draw_preemption_indicators(frame, tag_ts, btn_ts, uid_text, show_secs=2.0, uid_secs=3.0):
    now = time.time()
    y0 = 200
    if tag_ts and (now - tag_ts) < show_secs:
        cv2.putText(frame, "TAG ACIONADA!", (20, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80,255,80), 3)
    if btn_ts and (now - btn_ts) < show_secs:
        cv2.putText(frame, "BOTAO FISICO!", (20, y0+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,255,255), 3)
    if uid_text and tag_ts and (now - tag_ts) < uid_secs:
        cv2.putText(frame, f"UID: {uid_text}", (20, y0+80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,220,255), 2)

# ===================== SERIAL COM ESP32 =====================
SER_PORT = None  # ex.: "COM5" (Windows) ou "/dev/ttyUSB0" (Linux)
SER_BAUD = 115200

ser = None
rx_q = queue.Queue()

def find_esp_port():
    for p in serial.tools.list_ports.comports():
        name = (p.device or "") + " " + (p.description or "")
        if any(k in name.upper() for k in ["USB", "CP210", "CH340", "FTDI", "SILABS"]):
            return p.device
    return None

def serial_reader():
    global ser
    buf = b""
    while ser and ser.is_open:
        try:
            b = ser.read(1)
            if not b:
                continue
            if b in (b'\n', b'\r'):
                if buf:
                    line = buf.decode(errors="ignore").strip()
                    rx_q.put(line)
                    buf = b""
            else:
                buf += b
                if len(buf) > 256:
                    buf = b""
        except Exception:
            break

def serial_init():
    global ser, SER_PORT
    if SER_PORT is None:
        SER_PORT = find_esp_port() or ("COM5" if os.name == "nt" else "/dev/ttyUSB0")
    try:
        ser = serial.Serial(SER_PORT, SER_BAUD, timeout=0.01)
        th = threading.Thread(target=serial_reader, daemon=True)
        th.start()
        print(f"[SER] conectado em {SER_PORT}")
        return True
    except Exception as e:
        print(f"[SER] falha ao abrir {SER_PORT}: {e}")
        return False

def ser_send(line: str):
    if ser and ser.is_open:
        try:
            ser.write((line.strip() + "\n").encode())
        except Exception as e:
            print("[SER] erro envio:", e)

def outputs_to_bits(outs: dict) -> str:
    # Ordem: veh_g, veh_y, veh_r, ped_w, ped_dw
    return "".join([
        '1' if outs["veh_g"] else '0',
        '1' if outs["veh_y"] else '0',
        '1' if outs["veh_r"] else '0',
        '1' if outs["ped_w"] else '0',
        '1' if outs["ped_dw"] else '0',
    ])

# ===================== ARGUMENTOS CLI =====================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true",
                   help="Reduz latência descartando frames e usando inferência com frame reduzido")
    p.add_argument("reset", nargs="?", default=None,
                   help="Digite 'reset' para apagar a ROI e redesenhar")

    # presets de webcam
    p.add_argument("--480p", action="store_true", help="Força 640x480 @fps (padrão 30)")
    p.add_argument("--720p", action="store_true", help="Força 1280x720 @fps (padrão 30)")
    p.add_argument("--1080p", action="store_true", help="Força 1920x1080 @fps (padrão 30)")

    # custom
    p.add_argument("--width", type=int, help="Largura da webcam")
    p.add_argument("--height", type=int, help="Altura da webcam")
    p.add_argument("--fps", type=int, default=30, help="FPS desejado (padrão 30)")

    # porta serial opcional
    p.add_argument("--port", type=str, help="Porta serial do ESP32 (ex.: COM5, /dev/ttyUSB0)")
    return p.parse_args()

# ===================== MAIN =====================
def main():
    global tag_sim, SER_PORT

    args = parse_args()
    FAST = args.fast

    # reset ROI
    if args.reset and args.reset.lower() == "reset":
        if os.path.exists(ROI_JSON):
            os.remove(ROI_JSON)
            print("ROI apagada.")

    # Porta serial via CLI
    if args.port:
        SER_PORT = args.port

    # === Serial com ESP ===
    serial_ok = serial_init()
    if not serial_ok:
        print("[AVISO] Rodando sem ESP32 (sem saídas físicas).")

    # === YOLO ===
    model=YOLO(YOLO_MODEL_PATH)
    try: 
        model.fuse()
    except: 
        pass

    # === Câmera ===
    cap=cv2.VideoCapture(CAMERA_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

    # Se for webcam, aplique presets e custom
    is_webcam = isinstance(CAMERA_URL, int) or str(CAMERA_URL).isdigit()
    if is_webcam:# >>> Resolução manual preferencial <<<
        if MANUAL_RES:
            if MANUAL_RES == 480:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            elif MANUAL_RES == 720:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            elif MANUAL_RES == 1080:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            # Você pode acrescentar outros aqui
        if args.__dict__.get("480p"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, args.fps or 30)
        elif args.__dict__.get("720p"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, args.fps or 30)
        elif args.__dict__.get("1080p"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, args.fps or 30)
        if args.width and args.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if args.fps:
            cap.set(cv2.CAP_PROP_FPS, args.fps)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except:
            pass
        print("Webcam em:",
              int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              "x",
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              "@",
              int(cap.get(cv2.CAP_PROP_FPS)),
              "fps")

    if not cap.isOpened(): 
        print("Nao abriu camera"); 
        return

    tag_sim=TagSimulator()
    fsm=TrafficLightFSM()

    # Indicadores pré-empção no HUD
    last_tag_ts = None
    last_button_ts = None
    last_uid_text = None
    INDICATOR_SHOW_SECS = 2.0
    UID_SHOW_SECS = 3.0

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

    # Controle de envio para o ESP
    _last_bits = None
    _last_tx_ms = 0

    while True:
        # === leitura de frame ===
        if FAST:
            for _ in range(2): 
                cap.grab()
        ok,frame=cap.read()
        if not ok: 
            break

        draw_poly(frame,roi,C_ROI,fill=True)

        # === detecção apenas de pessoas ===
        infer_frame = frame if infer_scale==1.0 else cv2.resize(
            frame, None, fx=infer_scale, fy=infer_scale, interpolation=cv2.INTER_LINEAR)

        res = model(infer_frame, conf=DET_CONF, imgsz=imgsz, verbose=False)[0]
        pessoa_na_faixa=False

        if res.boxes is not None and len(res.boxes) > 0:
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

                color = C_PESSOA if inside else (120,220,120)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.circle(frame,(cx,cy),4,color,-1)

                if inside: 
                    pessoa_na_faixa=True

        # === RECEBE eventos do ESP (TAG/HB/PONG/BTN) ===
        try:
            while True:
                line = rx_q.get_nowait()
                if line.startswith("TAG,"):
                    uid = line.split(",",1)[1].strip()
                    fsm.request_tag()
                    last_tag_ts = time.time()
                    last_uid_text = uid
                elif line == "BTN,PED":
                    fsm.request_tag()
                    last_button_ts = time.time()
                elif line == "PONG":
                    pass
                elif line == "HB":
                    pass
        except queue.Empty:
            pass

        # === TAG simulação local (opcional) ===
        tag_sim.file_poll()
        pulse, sim_id = tag_sim.poll()
        if pulse: 
            fsm.request_tag()
            last_tag_ts = time.time()
            last_uid_text = f"SIM_{sim_id}" if sim_id is not None else "SIM"

        # === FSM e HUD ===
        fsm.update(pessoa_na_faixa)
        fsm.overlay(frame, pessoa_na_faixa)

        # Desenha o botão primeiro (fica por baixo)
        tag_sim.draw_button(frame)

        # Centraliza o semáforo no meio da tela
        h, w = frame.shape[:2]
        sem_w = 60 + 20 + 60   # largura: caixa carros (60) + gap (20) + caixa pedestre (60)
        sem_h = 160            # altura da caixa de carros (a maior)
        ox = w//2 - sem_w//2
        oy = 20
        draw_mini_traffic_lights(frame, fsm.outputs, origin_xy=(ox, oy))

        # Indicadores (texto) depois
        draw_preemption_indicators(frame, last_tag_ts, last_button_ts, last_uid_text,
                                show_secs=2.0, uid_secs=3.0)


        # === Envia saídas físicas ao ESP (quando mudam ou a cada 500ms) ===
        bits = outputs_to_bits(fsm.outputs)
        now_ms = int(time.time() * 1000)
        if serial_ok and (bits != _last_bits or (now_ms - _last_tx_ms) > 500):
            ser_send(f"OUT,{bits}")
            _last_bits = bits
            _last_tx_ms = now_ms

        # === janela e teclado (ordem correta: imshow -> waitKey) ===
        cv2.imshow(MAIN_WIN,frame)
        key=cv2.waitKey(1)&0xFF
        if key==ord('q') or key==27: 
            break
        tag_sim.keypress(key)

    # === finalizar ===
    cap.release(); 
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
