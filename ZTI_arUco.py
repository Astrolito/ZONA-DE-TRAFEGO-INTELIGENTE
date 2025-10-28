"""
smart_traffic_2zones_sim.py  (v3 - OpenCV 5+ ArUco OK)
Semáforo inteligente 2 zonas (webcam aérea) para maquete (carrinhos).

Recursos:
- YOLOv8 (veículos) + duas ROIs desenháveis (Zona A=NS, Zona B=LO)
- Verde pisca 1 Hz nos últimos 5s antes do amarelo
- Amarelo pisca 1 Hz (fase inteira)
- Modo EMERGENCIA (som e/ou ArUco ID=17):
    * Pausa tempos e memoriza fase/tempo restante
    * Eixo da emergência fica VERDE fixo
    * Eixo oposto ALTERNANDO EXCLUSIVO entre VERMELHO e AMARELO (1 Hz)
    * Banner "EMERGENCIA - NS/LO"
    * Ao sair, retoma exatamente de onde parou
- Tecla 'E' liga/desliga emergência manual (teste)

Instalação:
    pip install opencv-contrib-python ultralytics numpy
    # áudio opcional:
    pip install sounddevice

Como rodar:
    python smart_traffic_2zones_sim.py
    python smart_traffic_2zones_sim.py --fast --720p
    python smart_traffic_2zones_sim.py --mic
    python smart_traffic_2zones_sim.py reset   # redesenhar Zonas

Controles:
    q/ESC -> sair
    e     -> alterna EMERGENCIA manual
    Durante desenho das ROIs:
        ESQ:add | DIR:desfaz | MEIO:finalizar
"""

import os, time, json, argparse, threading, queue
from enum import Enum
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ===================== CONFIG =====================
YOLO_MODEL_PATH = "yolo11s.pt"
DEFAULT_CAMERA = 1
ROI_JSON = "roi_2zones.json"  # {"zoneA":[(x,y),...], "zoneB":[(x,y),...]}

VEHICLE_CLASSES = [2, 3, 5, 7]     # car, motorcycle, bus, truck
DET_CONF = 0.25
MIN_BBOX_AREA_FRAC = 1e-5
IMG_SZ = 640

# CORES
C_ZONEA = (50, 200, 255)
C_ZONEB = (255, 180, 60)
C_BOX   = (0, 255, 0)
C_BOX2  = (120, 220, 120)

MAIN_WIN = "Semaforo 2 Zonas (Simulacao)"

# ArUco (OpenCV 5+ API)
ARUCO_ID_EMER = 17
try:
    _aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    _aruco_params = cv2.aruco.DetectorParameters()
    _aruco_detector = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)
    HAVE_ARUCO = True
except Exception:
    HAVE_ARUCO = False
    _aruco_detector = None

# ===================== CLI =====================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true",
                   help="Reduz latência descartando frames e usando inferência menor")
    p.add_argument("reset", nargs="?", default=None,
                   help="Digite 'reset' para apagar as ROIs e redesenhar")

    # presets webcam
    p.add_argument("--480p", action="store_true")
    p.add_argument("--720p", action="store_true")
    p.add_argument("--1080p", action="store_true")

    # custom
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--camera", type=str)

    # áudio opcional
    p.add_argument("--mic", action="store_true", help="Detecção simples por som")

    return p.parse_args()

# ===================== ROI / UTILS =====================
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

# ===================== ÁUDIO SIMPLES =====================
class SimpleAudioTrigger:
    """
    Detector super simples de 'sinal alto' (RMS) por >= hold_s.
    Útil p/ maquete (teste). Para sirene real, depois plugamos classificador.
    """
    def __init__(self, enable: bool, rms_thr=0.08, hold_s=0.8):
        self.enabled = enable
        self.rms_thr = rms_thr
        self.hold_s = hold_s
        self._q = queue.Queue(maxsize=50)
        self._active = False
        self._last_on = 0.0
        self._thread = None
        self._stop = False

    def start(self):
        if not self.enabled: return
        try:
            import sounddevice as sd
        except Exception:
            print("[AUDIO] sounddevice não disponível. Instale: pip install sounddevice")
            self.enabled = False
            return

        def cb(indata, frames, time_info, status):
            if status: pass
            a = indata.astype(np.float32)
            if a.ndim > 1:
                a = a.mean(axis=1)
            rms = float(np.sqrt(np.mean(np.square(a)) + 1e-12))
            t = time.monotonic()
            try:
                self._q.put_nowait((t, rms))
            except queue.Full:
                pass

        def loop():
            import sounddevice as sd
            with sd.InputStream(callback=cb, channels=1, samplerate=16000, blocksize=2048):
                while not self._stop:
                    time.sleep(0.05)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self): self._stop = True

    def emergency(self) -> bool:
        if not self.enabled: return False
        now = time.monotonic()
        recent = []
        while True:
            try:
                recent.append(self._q.get_nowait())
            except queue.Empty:
                break
        recent = [x for x in recent if now - x[0] <= 1.5]
        if not recent: return False
        above = [t for (t, rms) in recent if rms >= self.rms_thr]
        if above and (max(above) - min(above) >= self.hold_s):
            self._last_on = now
            self._active = True
            return True
        if self._active and (now - self._last_on) < 1.0:
            return True
        self._active = False
        return False

# ===================== FSM =====================
class Phase(Enum):
    NS_GREEN=1
    NS_YEL=2
    ALL_RED=3
    LO_GREEN=4
    LO_YEL=5

MIN_GREEN = 5
BASE_GREEN = 20
YEL_TIME = 5
ALL_RED_TIME = 3
GREEN_EXT_STEP = 4
GREEN_MAX = 40

class TwoZoneController:
    def __init__(self):
        self.phase = Phase.NS_GREEN
        self.t0 = time.monotonic()
        self.tgt = BASE_GREEN
        self.outputs = dict(ns_g=True, ns_y=False, ns_r=False, lo_g=False, lo_y=False, lo_r=True)
        self._last_green_ns = True

    def _reset(self, dur):
        self.t0 = time.monotonic()
        self.tgt = float(dur)

    def _elapsed(self):
        return time.monotonic() - self.t0

    def remaining(self):
        return max(0.0, self.tgt - self._elapsed())

    def _outs(self, ns_g, ns_y, ns_r, lo_g, lo_y, lo_r):
        self.outputs = dict(ns_g=ns_g, ns_y=ns_y, ns_r=ns_r, lo_g=lo_g, lo_y=lo_y, lo_r=lo_r)

    def update(self, count_ns: int, count_lo: int):
        if self.phase == Phase.NS_GREEN:
            self._outs(True, False, False, False, False, True)
            if self._elapsed() >= self.tgt:
                if count_ns > 0 and (count_ns >= count_lo) and (self.tgt + GREEN_EXT_STEP <= GREEN_MAX):
                    self.tgt += GREEN_EXT_STEP
                else:
                    self.phase = Phase.NS_YEL
                    self._reset(YEL_TIME)
                    self._last_green_ns = True

        elif self.phase == Phase.NS_YEL:
            self._outs(False, True, False, False, False, True)
            if self._elapsed() >= self.tgt:
                self.phase = Phase.ALL_RED
                self._reset(ALL_RED_TIME)

        elif self.phase == Phase.ALL_RED:
            self._outs(False, False, True, False, False, True)
            if self._elapsed() >= self.tgt:
                if (count_ns > 0 or count_lo > 0) and (count_ns != count_lo):
                    if count_ns > count_lo:
                        self.phase = Phase.NS_GREEN
                    else:
                        self.phase = Phase.LO_GREEN
                    self._reset(max(MIN_GREEN, BASE_GREEN))
                else:
                    if self._last_green_ns:
                        self.phase = Phase.LO_GREEN
                    else:
                        self.phase = Phase.NS_GREEN
                    self._reset(MIN_GREEN)

        elif self.phase == Phase.LO_GREEN:
            self._outs(False, False, True, True, False, False)
            if self._elapsed() >= self.tgt:
                if count_lo > 0 and (count_lo >= count_ns) and (self.tgt + GREEN_EXT_STEP <= GREEN_MAX):
                    self.tgt += GREEN_EXT_STEP
                else:
                    self.phase = Phase.LO_YEL
                    self._reset(YEL_TIME)
                    self._last_green_ns = False

        elif self.phase == Phase.LO_YEL:
            self._outs(False, False, True, False, True, False)
            if self._elapsed() >= self.tgt:
                self.phase = Phase.ALL_RED
                self._reset(ALL_RED_TIME)

        return self.phase

    def overlay(self, frame, count_ns, count_lo):
        def put_right(y, text, scale=0.85, color=(255,255,255), th=2):
            (w,_),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
            x = frame.shape[1] - 20 - w
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), th+1)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th)

        phase_name = {
            Phase.NS_GREEN: "NS: VERDE",
            Phase.NS_YEL:   "NS: AMARELO (pisca)",
            Phase.ALL_RED:  "ALL RED",
            Phase.LO_GREEN: "LO: VERDE",
            Phase.LO_YEL:   "LO: AMARELO (pisca)",
        }[self.phase]

        priority = "NS" if count_ns > count_lo else ("LO" if count_lo > count_ns else "EMPATE")

        put_right(40, f"{phase_name} | T-{int(round(self.remaining()))}s")
        put_right(70,  f"Veiculos Zona A (NS): {count_ns}", 0.8, (200,255,200))
        put_right(95,  f"Veiculos Zona B (LO): {count_lo}", 0.8, (200,255,200))
        put_right(120, f"Prioridade (maior fila): {priority}", 0.8, (255,230,180))

# ===================== DESENHO =====================
def draw_traffic_lights(frame, outs, blink_ns_g=False, blink_ns_y=False,
                        blink_lo_g=False, blink_lo_y=False):
    hz = 1.0
    on = (int(time.monotonic() * hz) % 2) == 0

    def draw_light(x, y, on_r, on_y, on_g, label,
                   blink_g=False, blink_y=False):
        g = on_g
        y_ = on_y
        if blink_g: g = on_g and on
        if blink_y: y_ = on_y and on

        cv2.putText(frame, label, (x-5, y-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30,30,30), 2)
        cv2.circle(frame, (x, y),   12, (0,0,255)   if on_r else (60,60,60), -1)   # R
        cv2.circle(frame, (x, y+30),12, (0,255,255) if y_   else (60,60,60), -1)   # Y
        cv2.circle(frame, (x, y+60),12, (0,170,0)   if g    else (60,60,60), -1)   # G
        cv2.rectangle(frame, (x-20, y-22), (x+20, y+82), (80,80,80), 2)

    draw_light( 60, 50,
                outs["ns_r"], outs["ns_y"], outs["ns_g"], "NS",
                blink_g=blink_ns_g, blink_y=blink_ns_y)
    draw_light(140, 50,
                outs["lo_r"], outs["lo_y"], outs["lo_g"], "LO",
                blink_g=blink_lo_g, blink_y=blink_lo_y)

def draw_emergency_banner(frame, direction_text: str):
    h, w = frame.shape[:2]
    txt = f"EMERGENCIA - {direction_text}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    pad = 14
    x1 = max(10, (w - tw)//2 - 20)
    y1 = 10
    x2 = min(w-10, x1 + tw + 40)
    y2 = y1 + th + pad*2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (30,30,30), 2)
    cv2.putText(frame, txt, (x1+20, y1 + pad + th),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)

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

    # YOLO
    model = YOLO(YOLO_MODEL_PATH)
    try:
        model.fuse()
        _ = model.predict(np.zeros((IMG_SZ, IMG_SZ, 3), dtype=np.uint8),
                          conf=DET_CONF, imgsz=IMG_SZ, classes=VEHICLE_CLASSES,
                          verbose=False)
    except:
        pass

    # Áudio
    audio = SimpleAudioTrigger(enable=args.mic)
    audio.start()

    # Câmera
    camera = DEFAULT_CAMERA
    if args.camera:
        camera = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(camera)
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except: pass

    is_webcam = isinstance(camera, int) or (isinstance(camera, str) and camera.isdigit())
    if is_webcam:
        if args.__dict__.get("480p"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640);  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, args.fps or 30)
        elif args.__dict__.get("720p"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, args.fps or 30)
        elif args.__dict__.get("1080p"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, args.fps or 30)
        if args.width and args.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if args.fps:
            cap.set(cv2.CAP_PROP_FPS, args.fps)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except: pass

        print("Webcam em:",
              int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "@",
              int(cap.get(cv2.CAP_PROP_FPS)), "fps")

    if not cap.isOpened():
        print("Nao abriu camera")
        return

    # ROIs
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
                cv2.destroyWindow(window_title); break
            if key in (ord('q'), 27):
                cv2.destroyWindow(window_title); return None
        return poly

    if zoneA is None or zoneB is None:
        zoneA = acquire_zone("Defina Zona A (ex.: NS)")
        if zoneA is None: return
        zoneB = acquire_zone("Defina Zona B (ex.: LO)")
        if zoneB is None: return
        save_rois(zoneA, zoneB, ROI_JSON)

    # Máscaras
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    maskA = make_roi_mask((H, W, 3), zoneA)
    maskB = make_roi_mask((H, W, 3), zoneB)

    # Controlador
    ctl = TwoZoneController()

    # Emergência
    emergency_on = False
    emergency_dir: Optional[str] = None  # "NS" ou "LO"
    saved_phase = None
    saved_remaining = 0.0
    last_emerg_seen = 0.0

    infer_scale = 0.75 if FAST else 1.0
    min_area_abs = MIN_BBOX_AREA_FRAC * (H * W)

    def bbox_roi_overlap_ratio(x1,y1,x2,y2, mask):
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(mask.shape[1]-1, x2); y2 = min(mask.shape[0]-1, y2)
        if x2 <= x1 or y2 <= y1: return 0.0
        crop = mask[y1:y2, x1:x2]
        area_bbox = (x2-x1)*(y2-y1)
        if area_bbox <= 0: return 0.0
        area_in = int((crop > 0).sum())
        return area_in / float(area_bbox)

    while True:
        if FAST:
            for _ in range(2): cap.grab()
        ok, frame = cap.read()
        if not ok: break

        draw_poly(frame, zoneA, C_ZONEA, fill=True, alpha=0.18)
        draw_poly(frame, zoneB, C_ZONEB, fill=True, alpha=0.18)

        # Inferência
        infer_frame = frame if infer_scale == 1.0 else cv2.resize(
            frame, None, fx=infer_scale, fy=infer_scale, interpolation=cv2.INTER_LINEAR)

        res = model.predict(
            infer_frame, conf=DET_CONF, imgsz=IMG_SZ,
            classes=VEHICLE_CLASSES, verbose=False
        )[0]

        sx = frame.shape[1] / infer_frame.shape[1]
        sy = frame.shape[0] / infer_frame.shape[0]

        countA = 0
        countB = 0
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                if infer_scale != 1.0:
                    x1 = int(x1 * sx); x2 = int(x2 * sx)
                    y1 = int(y1 * sy); y2 = int(y2 * sy)

                area = max(1, (x2 - x1) * (y2 - y1))
                if area < min_area_abs: continue

                fracA = bbox_roi_overlap_ratio(x1,y1,x2,y2, maskA)
                fracB = bbox_roi_overlap_ratio(x1,y1,x2,y2, maskB)

                assigned = None
                if max(fracA, fracB) >= 0.06:
                    if fracA > fracB:
                        countA += 1; assigned = "A"
                    elif fracB > fracA:
                        countB += 1; assigned = "B"

                color = C_BOX if assigned in ("A","B") else C_BOX2
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            # === ArUco (emergência por tag) — ROI OBRIGATÓRIA (OpenCV 5+) ===
        aruco_dir = None
        if HAVE_ARUCO and _aruco_detector is not None:
            try:
                corners, ids, _ = _aruco_detector.detectMarkers(infer_frame)
                if ids is not None and len(ids) > 0:
                    ids = np.array(ids).flatten()
                    for i, cid in enumerate(ids):
                        if cid != ARUCO_ID_EMER:
                            continue

                        # pontos do marcador no frame de inferência (reduzido)
                        pts = corners[i][0].astype(np.float32)  # shape (4,2)

                        # centro convertido para o frame original
                        cx = int(np.mean(pts[:, 0]) * sx)
                        cy = int(np.mean(pts[:, 1]) * sy)

                        # (opcional) filtro de tamanho mínimo do marcador no frame original
                        # evita falsos positivos muito pequenos
                        pts_full_f = (pts * np.array([sx, sy], dtype=np.float32))
                        edges = [
                            np.linalg.norm(pts_full_f[0] - pts_full_f[1]),
                            np.linalg.norm(pts_full_f[1] - pts_full_f[2]),
                            np.linalg.norm(pts_full_f[2] - pts_full_f[3]),
                            np.linalg.norm(pts_full_f[3] - pts_full_f[0]),
                        ]
                        if max(edges) < 20:  # px mínimos, ajuste se quiser
                            continue

                        # desenho manual no frame original (sem drawDetectedMarkers)
                        pts_full = pts_full_f.astype(int)           # (4,2) int
                        cv2.polylines(frame, [pts_full.reshape(-1,1,2)],
                                    isClosed=True, color=(0,0,255), thickness=2)
                        for p in pts_full:
                            cv2.circle(frame, tuple(p), 3, (0,0,255), -1)
                        cv2.circle(frame, (cx, cy), 6, (0,255,0), -1)
                        cv2.putText(frame, f"ID {cid}", (cx-30, cy-15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                        # *** DISPARO SOMENTE SE ESTIVER DENTRO DA ROI ***
                        if 0 <= cy < maskA.shape[0] and 0 <= cx < maskA.shape[1]:
                            if maskA[cy, cx] > 0:
                                aruco_dir = "NS"   # Zona A
                                break
                            elif maskB[cy, cx] > 0:
                                aruco_dir = "LO"   # Zona B
                                break

                        # Se não caiu em nenhuma ROI: IGNORA (não seta aruco_dir)
                        # Nada de fallback para maior demanda.
            except Exception as e:
                print("[ARUCO ERR]", e)


        # Áudio → emergência
        audio_flag = audio.emergency()
        audio_dir = "NS" if (audio_flag and countA >= countB) else ("LO" if audio_flag else None)

        # Controla estado da emergência
        detected_dir = aruco_dir or audio_dir
        now = time.monotonic()
        if detected_dir:
            last_emerg_seen = now
            if not emergency_on:
                emergency_on = True
                emergency_dir = detected_dir
                saved_phase = ctl.phase
                saved_remaining = ctl.remaining()
            else:
                if aruco_dir:
                    emergency_dir = aruco_dir

        if emergency_on and (now - last_emerg_seen) > 2.0 and (not detected_dir):
            emergency_on = False
            ctl.phase = saved_phase
            ctl.t0 = time.monotonic() - (ctl.tgt - saved_remaining)
            emergency_dir = None

        # Atualiza FSM (pausado se emergência)
        if not emergency_on:
            ctl.update(countA, countB)

        # Pisca verde últimos 5s
        rem = ctl.remaining()
        blink_ns_g = (ctl.phase == Phase.NS_GREEN and rem <= 5.0 and not emergency_on)
        blink_lo_g = (ctl.phase == Phase.LO_GREEN and rem <= 5.0 and not emergency_on)
        # Pisca amarelo nas fases YEL
        blink_ns_y = (ctl.phase == Phase.NS_YEL and not emergency_on)
        blink_lo_y = (ctl.phase == Phase.LO_YEL and not emergency_on)

        # Saída base
        outs = ctl.outputs.copy()

        # Lógica visual da EMERGÊNCIA (alternância exclusiva Y/R no oposto)
        if emergency_on and emergency_dir:
            on = (int(time.monotonic() * 1.0) % 2) == 0  # 1 Hz
            if emergency_dir == "NS":
                # NS verde fixo; LO alterna entre R e Y
                outs["ns_g"], outs["ns_y"], outs["ns_r"] = True, False, False
                outs["lo_g"] = False
                outs["lo_y"], outs["lo_r"] = (True, False) if on else (False, True)
                blink_ns_g = blink_ns_y = blink_lo_g = blink_lo_y = False
            else:
                # LO verde fixo; NS alterna entre R e Y
                outs["lo_g"], outs["lo_y"], outs["lo_r"] = True, False, False
                outs["ns_g"] = False
                outs["ns_y"], outs["ns_r"] = (True, False) if on else (False, True)
                blink_ns_g = blink_ns_y = blink_lo_g = blink_lo_y = False

        # Desenha semáforos
        draw_traffic_lights(frame, outs,
                            blink_ns_g=blink_ns_g, blink_ns_y=blink_ns_y,
                            blink_lo_g=blink_lo_g, blink_lo_y=blink_lo_y)

        # HUD
        ctl.overlay(frame, countA, countB)

        # Banner
        if emergency_on:
            draw_emergency_banner(frame, "NS" if emergency_dir=="NS" else "LO")

        # Rodapé
        msg = f"Zona A=NS | Zona B=LO | Emergencia: {'ON' if emergency_on else 'OFF'} | Fonte: {'ArUco' if aruco_dir else ('Audio' if audio_flag else '---')}"
        cv2.putText(frame, msg, (20, frame.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,30,30), 2)
        cv2.putText(frame, msg, (20, frame.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1)

        # Janela/teclado
        cv2.imshow(MAIN_WIN, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('e'):
            if not emergency_on:
                emergency_on = True
                emergency_dir = "NS"
                saved_phase = ctl.phase
                saved_remaining = ctl.remaining()
                last_emerg_seen = time.monotonic()
            else:
                emergency_on = False
                ctl.phase = saved_phase
                ctl.t0 = time.monotonic() - (ctl.tgt - saved_remaining)
                emergency_dir = None

    audio.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        try: cv2.destroyAllWindows()
        except: pass
