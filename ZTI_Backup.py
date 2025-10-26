"""
smart_traffic_2zones_sim.py
Semáforo inteligente com 2 zonas (uma webcam aérea), otimizado para maquete (carrinhos).
- YOLOv8 (apenas veículos) + duas ROIs desenháveis (Zona A e Zona B)
- Extensão adaptativa de verde conforme presença/fila por zona
- Somente simulação visual no PC (sem ESP/serial, sem ArUco)

Instalação:
    pip install ultralytics opencv-python numpy

Como rodar:
    python smart_traffic_2zones_sim.py
    python smart_traffic_2zones_sim.py --fast
    python smart_traffic_2zones_sim.py --720p
    python smart_traffic_2zones_sim.py --camera 0
    python smart_traffic_2zones_sim.py reset   # redesenhar Zonas

Controles:
    q / ESC -> sair
    Na tela "Defina Zona A" e "Defina Zona B":
        Botão ESQ: adiciona ponto | Botão DIR: desfaz | Botão MEIO: finalizar
"""

import os, time, json, argparse
from enum import Enum
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ===================== CONFIG BÁSICA =====================
YOLO_MODEL_PATH = "yolo11s.pt"

# Webcam/stream
DEFAULT_CAMERA = "video3.mp4"  # índice ou URL (RTSP/HTTP)

# ROI (duas zonas)
ROI_JSON = "roi_2zones.json"  # {"zoneA":[(x,y),...], "zoneB":[(x,y),...]}

# Classes COCO para veículos: 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 3, 5, 7]
DET_CONF = 0.25           # ligeiramente baixo por serem objetos pequenos
MIN_BBOX_AREA_FRAC = 1e-5 # min área do bbox vs área do frame (para carrinhos)
IMG_SZ = 240

# Cores
C_ZONEA = (50, 200, 255)
C_ZONEB = (255, 180, 60)
C_BOX   = (0, 255, 0)
C_BOX2  = (120, 220, 120)

MAIN_WIN = "Semaforo 2 Zonas (Simulacao)"

# ===================== ARGUMENTOS CLI =====================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true",
                   help="Reduz latência descartando frames e usando inferência com frame reduzido")
    p.add_argument("reset", nargs="?", default=None,
                   help="Digite 'reset' para apagar as ROIs e redesenhar")

    # presets de webcam
    p.add_argument("--480p", action="store_true", help="Força 640x480 @fps (padrão 30)")
    p.add_argument("--720p", action="store_true", help="Força 1280x720 @fps (padrão 30)")
    p.add_argument("--1080p", action="store_true", help="Força 1920x1080 @fps (padrão 30)")

    # custom
    p.add_argument("--width", type=int, help="Largura da webcam")
    p.add_argument("--height", type=int, help="Altura da webcam")
    p.add_argument("--fps", type=int, default=30, help="FPS desejado (padrão 30)")
    p.add_argument("--camera", type=str, help="Índice da webcam (ex.: 0/1/2) ou URL RTSP/HTTP")

    return p.parse_args()

# ===================== ROI E UTILS =====================
_clicked: List[Tuple[int,int]]=[]; _selecting=False

def save_rois(zoneA, zoneB, path):
    with open(path, "w") as f:
        f.write(json.dumps({"zoneA": zoneA, "zoneB": zoneB}))

def load_rois(path):
    with open(path, "r") as f:
        data = json.load(f)
    a = [tuple(p) for p in data["zoneA"]]
    b = [tuple(p) for p in data["zoneB"]]
    return a, b

def draw_poly(frame, poly, color, fill=True, alpha=0.2):
    if not poly: return
    pts = np.array(poly, np.int32)
    if fill:
        ov = frame.copy()
        cv2.fillPoly(ov, [pts], color)
        frame[:] = cv2.addWeighted(ov, alpha, frame, 1.0 - alpha, 0)
    cv2.polylines(frame, [pts], True, color, 2)

def in_poly(pt, poly): 
    return cv2.pointPolygonTest(np.array(poly, np.int32), pt, False) >= 0

def make_roi_mask(shape, poly):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if poly:
        cv2.fillPoly(mask, [np.array(poly, np.int32)], 255)
    return mask

def mouse_cb(ev, x, y, flags, param):
    global _clicked, _selecting
    if ev == cv2.EVENT_LBUTTONDOWN:
        _clicked.append((x,y)); _selecting=True
    elif ev == cv2.EVENT_RBUTTONDOWN and _clicked:
        _clicked.pop()
    elif ev == cv2.EVENT_MBUTTONDOWN:
        _selecting=False

# ===================== FSM 2 ZONAS =====================
class Phase(Enum):
    NS_GREEN=1
    NS_YEL=2
    ALL_RED=3
    EW_GREEN=4
    EW_YEL=5

# Tempos (s) — ajuste para maquete
MIN_GREEN = 5           # verde mínimo antes de troca
BASE_GREEN = 20         # verde base quando há demanda
YEL_TIME = 5            # amarelo
ALL_RED_TIME = 3        # intertravamento
GREEN_EXT_STEP = 4      # extensão quando há fila
GREEN_MAX = 40          # teto de verde para evitar starvation

class TwoZoneController:
    """
    Fases:
      NS_GREEN → NS_YEL → ALL_RED → EW_GREEN → EW_YEL → ALL_RED → loop
    - Demanda: presença de veículos em cada zona
    - Extensão: enquanto houver fila na zona atual, até GREEN_MAX
    Saídas: ns_g, ns_y, ns_r, ew_g, ew_y, ew_r
    """
    def __init__(self):
        self.phase = Phase.NS_GREEN
        self.t0 = time.monotonic()
        self.tgt = BASE_GREEN
        self.outputs = dict(ns_g=True, ns_y=False, ns_r=False, ew_g=False, ew_y=False, ew_r=True)

    def _reset(self, dur):
        self.t0 = time.monotonic()
        self.tgt = dur

    def _elapsed(self):
        return time.monotonic() - self.t0

    def remaining(self):
        return max(0.0, self.tgt - self._elapsed())

    def _outs(self, ns_g, ns_y, ns_r, ew_g, ew_y, ew_r):
        self.outputs = dict(ns_g=ns_g, ns_y=ns_y, ns_r=ns_r, ew_g=ew_g, ew_y=ew_y, ew_r=ew_r)

    def update(self, demand_ns: bool, demand_ew: bool):
        # NS GREEN
        if self.phase == Phase.NS_GREEN:
            self._outs(True, False, False, False, False, True)
            if self._elapsed() >= self.tgt:
                if demand_ns and (self.tgt + GREEN_EXT_STEP <= GREEN_MAX):
                    self.tgt += GREEN_EXT_STEP
                else:
                    self.phase = Phase.NS_YEL; self._reset(YEL_TIME)

        # NS YELLOW
        elif self.phase == Phase.NS_YEL:
            self._outs(False, True, False, False, False, True)
            if self._elapsed() >= self.tgt:
                self.phase = Phase.ALL_RED; self._reset(ALL_RED_TIME)

        # ALL RED
        elif self.phase == Phase.ALL_RED:
            self._outs(False, False, True, False, False, True)
            if self._elapsed() >= self.tgt:
                # prioridade: quem tem demanda
                if demand_ns and not demand_ew:
                    self.phase = Phase.NS_GREEN; self._reset(max(BASE_GREEN, MIN_GREEN))
                elif demand_ew and not demand_ns:
                    self.phase = Phase.EW_GREEN; self._reset(max(BASE_GREEN, MIN_GREEN))
                else:
                    # empates/sem demanda: alterna
                    self.phase = Phase.EW_GREEN if self.outputs["ns_r"] else Phase.NS_GREEN
                    self._reset(MIN_GREEN)

        # EW GREEN
        elif self.phase == Phase.EW_GREEN:
            self._outs(False, False, True, True, False, False)
            if self._elapsed() >= self.tgt:
                if demand_ew and (self.tgt + GREEN_EXT_STEP <= GREEN_MAX):
                    self.tgt += GREEN_EXT_STEP
                else:
                    self.phase = Phase.EW_YEL; self._reset(YEL_TIME)

        # EW YELLOW
        elif self.phase == Phase.EW_YEL:
            self._outs(False, False, True, False, True, False)
            if self._elapsed() >= self.tgt:
                self.phase = Phase.ALL_RED; self._reset(ALL_RED_TIME)

        return self.phase

    def overlay(self, frame, demand_ns, demand_ew):
        def put_right(y, text, scale=0.85, color=(255,255,255), th=2):
            (w,_),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
            x = frame.shape[1] - 20 - w
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), th+1)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th)

        phase_name = {
            Phase.NS_GREEN: "NS: VERDE",
            Phase.NS_YEL:   "NS: AMARELO",
            Phase.ALL_RED:  "ALL RED",
            Phase.EW_GREEN: "EW: VERDE",
            Phase.EW_YEL:   "EW: AMARELO",
        }[self.phase]

        put_right(40, f"{phase_name} | T-{int(round(self.remaining()))}s")
        put_right(70, f"Demanda NS: {'SIM' if demand_ns else 'NAO'}", 0.8, (200,255,200))
        put_right(95, f"Demanda EW: {'SIM' if demand_ew else 'NAO'}", 0.8, (200,255,200))

# ===================== DESENHO DOS SEMÁFOROS NA TELA =====================
def draw_traffic_lights(frame, outs):
    # blocos NS e EW no canto superior esquerdo
    def draw_light(x, y, on_r, on_y, on_g, label):
        cv2.putText(frame, label, (x-5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,30,30), 2)
        cv2.circle(frame, (x, y),   10, (0,0,255) if on_r else (60,60,60), -1)
        cv2.circle(frame, (x, y+25),10, (0,255,255) if on_y else (60,60,60), -1)
        cv2.circle(frame, (x, y+50),10, (0,170,0)   if on_g else (60,60,60), -1)
        cv2.rectangle(frame, (x-18, y-18), (x+18, y+68), (80,80,80), 2)

    draw_light(40, 40,  outs["ns_r"], outs["ns_y"], outs["ns_g"], "NS")
    draw_light(100, 40, outs["ew_r"], outs["ew_y"], outs["ew_g"], "EW")

# ===================== MAIN =====================
def main():
    global IMG_SZ

    args = parse_args()
    FAST = args.fast

    # reset ROIs
    if args.reset and args.reset.lower() == "reset":
        if os.path.exists(ROI_JSON):
            os.remove(ROI_JSON)
            print("ROIs apagadas.")

    # === YOLO ===
    model = YOLO(YOLO_MODEL_PATH)
    try:
        model.fuse()
        # Warmup para estabilizar latência
        _ = model.predict(np.zeros((IMG_SZ, IMG_SZ, 3), dtype=np.uint8),
                          conf=DET_CONF, imgsz=IMG_SZ, classes=VEHICLE_CLASSES,
                          verbose=False)
    except:
        pass

    # === Câmera ===
    camera = DEFAULT_CAMERA
    if args.camera:
        camera = int(args.camera) if args.camera.isdigit() else args.camera

    cap = cv2.VideoCapture(camera)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except:
        pass

    is_webcam = isinstance(camera, int) or (isinstance(camera, str) and camera.isdigit())
    if is_webcam:
        if args.__dict__.get("480p"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, args.fps or 30)
        elif args.__dict__.get("720p"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set=cv2.CAP_PROP_FPS
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
        print("Nao abriu camera")
        return

    # === ROIs ===
    if os.path.exists(ROI_JSON):
        try:
            zoneA, zoneB = load_rois(ROI_JSON)
            print("ROIs carregadas.")
        except Exception as e:
            print("Falha ao ler ROIs, redesenhando...", e)
            zoneA, zoneB = None, None
    else:
        zoneA, zoneB = None, None

    def acquire_zone(window_title):
        global _clicked, _selecting
        _clicked, _selecting = [], False
        cv2.namedWindow(window_title)
        cv2.setMouseCallback(window_title, mouse_cb)
        poly = None
        while True:
            for _ in range(2): cap.grab()
            ok, frame = cap.read()
            if not ok: break
            prev = frame.copy()
            for p in _clicked:
                cv2.circle(prev, p, 4, (0,255,255), -1)
            if len(_clicked) >= 2:
                cv2.polylines(prev, [np.array(_clicked, np.int32)], False, (0,255,255), 2)
            cv2.putText(prev, "ESQ:add | DIR:desfaz | MEIO:finalizar", (20,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,255,255), 2)
            cv2.imshow(window_title, prev)
            key = cv2.waitKey(1) & 0xFF
            if not _selecting and len(_clicked) >= 3:
                poly = _clicked.copy()
                cv2.destroyWindow(window_title)
                break
            if key in (ord('q'), 27):
                cv2.destroyWindow(window_title)
                return None
        return poly

    if zoneA is None or zoneB is None:
        zoneA = acquire_zone("Defina Zona A (ex.: NS)")
        if zoneA is None: return
        zoneB = acquire_zone("Defina Zona B (ex.: EW)")
        if zoneB is None: return
        save_rois(zoneA, zoneB, ROI_JSON)

    # Pré-compute máscaras das zonas
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    maskA = make_roi_mask((H, W, 3), zoneA)
    maskB = make_roi_mask((H, W, 3), zoneB)

    # === Controlador ===
    ctl = TwoZoneController()

    # Inferência
    infer_scale = 0.75 if FAST else 1.0
    min_area_abs = MIN_BBOX_AREA_FRAC * (H * W)

    def bbox_roi_overlap(x1,y1,x2,y2, mask, thr=0.06):
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(mask.shape[1]-1, x2); y2 = min(mask.shape[0]-1, y2)
        if x2 <= x1 or y2 <= y1: return False
        crop = mask[y1:y2, x1:x2]
        area_bbox = (x2-x1)*(y2-y1)
        if area_bbox <= 0: return False
        area_in = int((crop > 0).sum())
        return (area_in / float(area_bbox)) >= thr

    while True:
        if FAST:
            for _ in range(2): cap.grab()
        ok, frame = cap.read()
        if not ok: break

        draw_poly(frame, zoneA, C_ZONEA, fill=True, alpha=0.18)
        draw_poly(frame, zoneB, C_ZONEB, fill=True, alpha=0.18)

        # Frame de inferência
        infer_frame = frame if infer_scale == 1.0 else cv2.resize(
            frame, None, fx=infer_scale, fy=infer_scale, interpolation=cv2.INTER_LINEAR)

        # === YOLO: veículos ===
        res = model.predict(
            infer_frame,
            conf=DET_CONF,
            imgsz=IMG_SZ,
            classes=VEHICLE_CLASSES,
            verbose=False
        )[0]

        sx = frame.shape[1] / infer_frame.shape[1]
        sy = frame.shape[0] / infer_frame.shape[0]

        demandA = False
        demandB = False

        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                if infer_scale != 1.0:
                    x1 = int(x1 * sx); x2 = int(x2 * sx)
                    y1 = int(y1 * sy); y2 = int(y2 * sy)

                area = max(1, (x2 - x1) * (y2 - y1))
                if area < min_area_abs:
                    continue

                insideA = bbox_roi_overlap(x1,y1,x2,y2, maskA, thr=0.06)
                insideB = bbox_roi_overlap(x1,y1,x2,y2, maskB, thr=0.06)

                color = C_BOX2
                if insideA or insideB: color = C_BOX
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

                if insideA: demandA = True
                if insideB: demandB = True

        # === Atualiza controlador e HUD ===
        ctl.update(demandA, demandB)
        ctl.overlay(frame, demandA, demandB)

        # Desenho dos semáforos simulados
        draw_traffic_lights(frame, ctl.outputs)

        # Legenda rápida
        cv2.putText(frame, "Zona A = NS | Zona B = EW | (Sem ArUco / Emergencia)", 
                    (20, frame.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,30,30), 2)
        cv2.putText(frame, "Zona A = NS | Zona B = EW | (Sem ArUco / Emergencia)", 
                    (20, frame.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1)

        # === janela/teclado ===
        cv2.imshow(MAIN_WIN, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        try:
            cv2.destroyAllWindows()
        except:
            pass
