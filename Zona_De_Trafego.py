"""
two_signal_intersection.py
Cruzamento com dois semáforos sincronizados:
- Duas fontes de vídeo (ou a mesma para testes)
- Conta veículos por aproximação (ROI retangular por câmera)
- O lado com mais carros ganha MAIS TEMPO de verde (extensão dinâmica)
- Preempção por emergência: carro OU caminhão vermelhos em um lado fecham o outro

Instalação:
    pip install ultralytics opencv-python numpy

Como rodar:
    python two_signal_intersection.py

Controles:
    q -> sair

Ajustes rápidos:
- CAM_A_URL / CAM_B_URL (RTSP ou webcams 0/1)
- COUNT_ROI_A / COUNT_ROI_B (x1,y1,x2,y2) – região onde os carros se acumulam
"""

import cv2, time
import numpy as np
from enum import Enum
from ultralytics import YOLO

# ===================== CONFIG FONTE =====================
# Use duas câmeras IP, ou a mesma para teste (defina CAM_B_URL = CAM_A_URL), ou webcams 0/1.
CAM_A_URL = "rtsp://admin:QT9RJ462@192.168.0.108:554/cam/realmonitor?channel=1&subtype=1"
CAM_B_URL = 0 #"rtsp://admin:Naoesquecer1@192.168.100.7:554/cam/realmonitor?channel=1&subtype=1"
# CAM_A_URL, CAM_B_URL = 0, 1         # exemplo com duas webcams
# CAM_B_URL = CAM_A_URL               # para testar com uma única fonte

# ROI de contagem (x1,y1,x2,y2) em cada câmera
COUNT_ROI_A = (100, 200, 800, 700)
COUNT_ROI_B = (100, 200, 800, 700)

# ===================== CONFIG DETECÇÃO =====================
YOLO_MODEL_PATH = "yolov8n.pt"
DET_CONF = 0.5
CL_VEIC = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# HSV para vermelho (emergência)
RED_RANGES = [
    (np.array([0, 80, 80]),   np.array([10, 255, 255])),
    (np.array([160, 80, 80]), np.array([179, 255, 255])),
]
RED_RATIO_THR = 0.12  # >= 12% dos pixels no bbox em vermelho => vermelho

# ===================== TEMPOS DO CICLO =====================
MIN_GREEN   = 30   # verde mínimo de qualquer lado
YELLOW_TIME = 5
ALL_RED     = 2
MAX_GREEN   = 60   # verde máximo para não travar o outro lado
REVISIT_PERIOD = 0.5  # frequência (s) de reavaliação

# Extensão de verde em função da diferença de carros:
# verde_alvo = clamp(MIN_GREEN + GAIN_PER_CAR * (cnt_active - cnt_other), MIN_GREEN, MAX_GREEN)
GAIN_PER_CAR = 2.0  # segundos extras por carro a mais no lado ativo

# ===================== CORES / JANELAS =====================
C_ROI = (255, 0, 255)
C_V   = (255, 255, 0)
C_E   = (0, 0, 255)

WIN_A = "A - Aproximação"
WIN_B = "B - Aproximação"
MAIN  = "Cruzamento (sincronizado)"

# ===================== FUNÇÕES AUXILIARES =====================
def red_ratio(bgr, x1, y1, x2, y2):
    h, w = bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w-1, x2); y2 = min(h-1, y2)
    if x2 <= x1 or y2 <= y1: return 0.0
    crop = bgr[y1:y2, x1:x2]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    masks = [cv2.inRange(hsv, lo, hi) for lo, hi in RED_RANGES]
    m = masks[0]
    for k in masks[1:]:
        m = cv2.bitwise_or(m, k)
    return float(np.count_nonzero(m)) / float(m.size)

def count_and_emergency(model, frame, roi_rect):
    x1, y1, x2, y2 = roi_rect
    H, W = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
    roi = frame[y1:y2, x1:x2].copy()

    res = model(roi, conf=DET_CONF)[0]
    count = 0; emergency = False

    if res.boxes is not None:
        names = res.names if hasattr(res, "names") else {}
        for b in res.boxes:
            cls = int(b.cls[0]); conf = float(b.conf[0])
            if cls not in CL_VEIC: continue
            bx1, by1, bx2, by2 = map(int, b.xyxy[0])

            rr = red_ratio(roi, bx1, by1, bx2, by2)
            is_red = rr >= RED_RATIO_THR
            # >>> caminhões vermelhos também são emergência (cls==7)
            is_emergency = is_red and (cls == 2 or cls == 7)
            if is_emergency: emergency = True

            count += 1
            color = C_E if is_emergency else C_V
            cv2.rectangle(roi, (bx1, by1), (bx2, by2), color, 2)
            lbl = names.get(cls, str(cls))
            cv2.putText(roi, f"{lbl} {conf:.2f}", (bx1, by1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # borda da ROI (só para visual)
    cv2.rectangle(roi, (0, 0), (roi.shape[1]-1, roi.shape[0]-1), C_ROI, 2)
    return count, emergency, roi

# ===================== CONTROLADOR DE CRUZAMENTO =====================
class Side(Enum):
    A = 1
    B = 2

class Phase(Enum):
    GREEN  = 1
    YELLOW = 2
    ALLRED = 3

class IntersectionController:
    def __init__(self):
        self.active = Side.A
        self.phase  = Phase.GREEN
        self.t0     = time.monotonic()
        self.tgt    = MIN_GREEN
        self.max_t0 = time.monotonic()  # início do verde atual (para MAX_GREEN)
        self.emergA = False
        self.emergB = False

    def _reset_timer(self, d):
        self.t0 = time.monotonic()
        self.tgt = d

    def _elapsed(self):
        return time.monotonic() - self.t0

    def _max_elapsed(self):
        return time.monotonic() - self.max_t0

    def remaining(self):
        return max(0.0, self.tgt - self._elapsed())

    def set_emergency(self, a: bool, b: bool):
        self.emergA = a; self.emergB = b

    def _apply_dynamic_green_target(self, cnt_active, cnt_other):
        """
        Extende dinamicamente o alvo do GREEN do lado ativo com base na diferença de contagem.
        Não reduz o alvo no meio do verde (só aumenta). Sempre respeita MIN_GREEN..MAX_GREEN.
        """
        diff = max(0, cnt_active - cnt_other)  # só estende se ativo está mais carregado
        desired = MIN_GREEN + GAIN_PER_CAR * diff
        desired = max(MIN_GREEN, min(MAX_GREEN, desired))
        # só aumenta o alvo; não encurta durante o ciclo
        if desired > self.tgt:
            # evita pulo brusco: garante que sempre haja pelo menos 1s restante
            self.tgt = max(desired, self._elapsed() + 1.0)

    def decide(self, cntA: int, cntB: int):
        """
        Regras:
        - Emergência: se A tem (carro/caminhão vermelho) e B não, trocar para A; vice-versa.
        - Extensão dinâmica: lado ativo com mais carros ganha mais tempo de green.
        - Troca por carga: após MIN_GREEN, se o outro lado tiver mais carros, preparar troca.
        - MAX_GREEN: ao atingir, troca mesmo que o ativo ainda esteja mais carregado.
        """
        # 1) Preempção por emergência (corta após >1s de green para evitar bounce instantâneo)
        if self.phase == Phase.GREEN:
            if self.emergA and not self.emergB and self.active != Side.A and self._elapsed() >= 1.0:
                self.phase = Phase.YELLOW; self._reset_timer(YELLOW_TIME); self.next_side = Side.A; return
            if self.emergB and not self.emergA and self.active != Side.B and self._elapsed() >= 1.0:
                self.phase = Phase.YELLOW; self._reset_timer(YELLOW_TIME); self.next_side = Side.B; return

        # 2) Extensão dinâmica do verde do lado ativo
        if self.phase == Phase.GREEN:
            if self.active == Side.A:
                self._apply_dynamic_green_target(cntA, cntB)
            else:
                self._apply_dynamic_green_target(cntB, cntA)

        # 3) Lógica de troca regular (carga e limites)
        if self.phase == Phase.GREEN:
            min_done = self._elapsed() >= MIN_GREEN
            max_done = self._max_elapsed() >= MAX_GREEN

            # Se estourou green máximo, força troca
            if max_done:
                self.phase = Phase.YELLOW; self._reset_timer(YELLOW_TIME)
                self.next_side = Side.B if self.active == Side.A else Side.A
                return

            # Se já cumpriu o mínimo e o outro lado está mais carregado, troca
            if min_done:
                if (self.active == Side.A and cntB > cntA) or (self.active == Side.B and cntA > cntB):
                    # só troca se já alcançou o alvo atual (para não interromper a extensão recém-ajustada)
                    if self._elapsed() >= self.tgt:
                        self.phase = Phase.YELLOW; self._reset_timer(YELLOW_TIME)
                        self.next_side = Side.B if self.active == Side.A else Side.A

        elif self.phase == Phase.YELLOW:
            if self._elapsed() >= self.tgt:
                self.phase = Phase.ALLRED; self._reset_timer(ALL_RED)

        elif self.phase == Phase.ALLRED:
            if self._elapsed() >= self.tgt:
                if hasattr(self, "next_side"):
                    self.active = self.next_side
                    delattr(self, "next_side")
                # inicia novo GREEN
                self.phase = Phase.GREEN
                self._reset_timer(MIN_GREEN)
                self.max_t0 = time.monotonic()  # reinicia contador do green máximo

    def lamps(self):
        """Estados das 'lâmpadas' simuladas para cada lado."""
        if self.phase == Phase.GREEN:
            if self.active == Side.A:
                return dict(A_GREEN=True, A_YEL=False, A_RED=False,
                            B_GREEN=False, B_YEL=False, B_RED=True)
            else:
                return dict(A_GREEN=False, A_YEL=False, A_RED=True,
                            B_GREEN=True,  B_YEL=False, B_RED=False)
        elif self.phase == Phase.YELLOW:
            if self.active == Side.A:
                return dict(A_GREEN=False, A_YEL=True, A_RED=False,
                            B_GREEN=False, B_YEL=False, B_RED=True)
            else:
                return dict(A_GREEN=False, A_YEL=False, A_RED=True,
                            B_GREEN=False, B_YEL=True,  B_RED=False)
        else:  # ALLRED
            return dict(A_GREEN=False, A_YEL=False, A_RED=True,
                        B_GREEN=False, B_YEL=False, B_RED=True)

    def overlay(self, canvas, cntA, cntB):
        h, w = canvas.shape[:2]
        def put(text, y, scale=0.8, color=(255,255,255)):
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            x = w - 20 - tw
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3)
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

        state = f"ATIVO: {'A' if self.active==Side.A else 'B'} | FASE: {self.phase.name} | T-{int(self.remaining())}s"
        put(state, 30)
        # use color= para não confundir com scale
        put(f"cntA={cntA}  emergA={'SIM' if self.emergA else 'NAO'}", 60, color=(200,255,200))
        put(f"cntB={cntB}  emergB={'SIM' if self.emergB else 'NAO'}", 90, color=(200,255,200))
        lamps = self.lamps()
        put("Lampadas: " + " ".join([k for k, v in lamps.items() if v]), 120)

# ===================== MAIN =====================
def open_cap(src):
    if isinstance(src, tuple): return cv2.VideoCapture(*src)
    return cv2.VideoCapture(src)

def draw_roi_rect(frame, rect, color):
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

def bulb(img, center, on, color):
    cv2.circle(img, center, 18, (40,40,40), -1)
    if on: cv2.circle(img, center, 16, color, -1)

def main():
    model = YOLO(YOLO_MODEL_PATH)

    capA = open_cap(CAM_A_URL); capA.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capB = open_cap(CAM_B_URL); capB.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capA.isOpened() or not capB.isOpened():
        print("Nao abriu uma das fontes de video."); return

    ctrl = IntersectionController()
    last_decide = time.time()

    while True:
        okA, frmA = capA.read()
        okB, frmB = capB.read()
        if not okA or not okB: break

        # Contagem e emergência por aproximação (carro/caminhão vermelhos detonam emergência)
        cntA, emergA, roiA = count_and_emergency(model, frmA, COUNT_ROI_A)
        cntB, emergB, roiB = count_and_emergency(model, frmB, COUNT_ROI_B)

        # Visual das ROIs nas imagens cruas
        draw_roi_rect(frmA, COUNT_ROI_A, C_ROI)
        draw_roi_rect(frmB, COUNT_ROI_B, C_ROI)

        # Janelas individuais das aproximações
        cv2.imshow(WIN_A, roiA)
        cv2.imshow(WIN_B, roiB)

        # Alimenta estados de emergência
        ctrl.set_emergency(emergA, emergB)

        # Decide em intervalos regulares
        if (time.time() - last_decide) >= REVISIT_PERIOD:
            ctrl.decide(cntA, cntB)
            last_decide = time.time()

        # Canvas principal
        canvas = np.zeros((190, 660, 3), dtype=np.uint8)
        ctrl.overlay(canvas, cntA, cntB)

        # “Lâmpadas” simuladas
        lamps = ctrl.lamps()
        bulb(canvas, (60,150),  lamps["A_GREEN"], (0,255,0))
        bulb(canvas, (100,150), lamps["A_YEL"],   (0,255,255))
        bulb(canvas, (140,150), lamps["A_RED"],   (0,0,255))
        cv2.putText(canvas, "A", (155,156), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        bulb(canvas, (240,150), lamps["B_GREEN"], (0,255,0))
        bulb(canvas, (280,150), lamps["B_YEL"],   (0,255,255))
        bulb(canvas, (320,150), lamps["B_RED"],   (0,0,255))
        cv2.putText(canvas, "B", (335,156), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow(MAIN, canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capA.release(); capB.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
