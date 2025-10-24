"""
ped_crosswalk_tag_audio_vu.py
Faixa de pedestres com:
- YOLOv8 (pessoas + veículos) e ROI da faixa
- Simulador de TAG (teclado 't'/'1'..'9', botão na tela, arquivo tag.trigger)
- FSM (semáforo): VEH_GREEN → VEH_YEL → ALL_RED_TO_PED → PED_GREEN → ALL_RED_TO_VEH → VEH_GREEN
- Áudio RTSP da câmera IM4C via PyAV: medidor de volume (VU) em dB e detecção simples de sirene
- HUD mostra estados (carros/pedestre), pessoa na faixa, nível de áudio e sirene

Instalação rápida (PC):
    pip install ultralytics opencv-python numpy av

Como rodar:
    python ped_crosswalk_tag_audio_vu.py
    python ped_crosswalk_tag_audio_vu.py reset   # apaga ROI e redesenha

Controles:
    q / Q   -> sair
    t       -> simula TAG
    1..9    -> simula TAG com ID
    Clique no botão "TAG" na tela
    touch/tag.trigger -> dispara uma vez (crie o arquivo "tag.trigger" na pasta do script)
"""

import sys, os, time, json, threading, collections, math
from enum import Enum
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ===================== CONFIG BÁSICA =====================
YOLO_MODEL_PATH = "yolov8n.pt"

# RTSP da sua IM4C (habilite ÁUDIO no webconfig e use AAC ou G.711):
CAMERA_URL = "rtsp://admin:QT9RJ462@192.168.0.108:554/cam/realmonitor?channel=1&subtype=1"
#CAMERA_URL = 0  # descomente para usar webcam (sem áudio via RTSP)

ROI_JSON = "roi_pedestre.json"

# Classes COCO
CL_PESSOA  = {0}
CL_VEICULO = {2, 3, 5, 7}
DET_CONF   = 0.5

# Cores
C_PESSOA  = (0,255,0)
C_VEICULO = (255,255,0)
C_ROI     = (255,0,255)

MAIN_WIN = "Faixa Pedestre + TAG + Áudio (VU/Sirene)"

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
VEH_GREEN_DEFAULT=20
VEH_YEL_TIME=5
ALL_RED_CLEAR=3
PED_GREEN_BASE=12
PED_EXT_STEP=5
PEDESTRIAN_CLEAR_GRACE=5
PED_ENTRY_BOOST=5

# Preempção por TAG (acelera fechamento do verde dos carros)
MIN_GREEN_BEFORE_PREEMPT=5
PREEMPT_CUT_TO=5

class TrafficLightFSM:
    def __init__(self):
        self.state=State.VEH_GREEN
        self.t0=time.monotonic(); self.tgt=VEH_GREEN_DEFAULT
        self.tag_pending=False
        self.last_seen_person_ts=None
        self.outputs=dict(veh_g=True,veh_y=False,veh_r=False,ped_w=False,ped_dw=True)

    def _reset(self,dur): self.t0=time.monotonic(); self.tgt=dur
    def _elapsed(self): return time.monotonic()-self.t0
    def remaining(self): return max(0.0,self.tgt-self._elapsed())
    def request_tag(self): self.tag_pending=True
    def _outs(self,vg,vy,vr,pw,pdw): self.outputs=dict(veh_g=vg,veh_y=vy,veh_r=vr,ped_w=pw,ped_dw=pdw)

    def update(self,pessoa_na_faixa:bool):
        now=time.monotonic()

        # TAG durante VERDE carros => corta p/ amarelo
        if self.state==State.VEH_GREEN and self.tag_pending:
            if self._elapsed()>=MIN_GREEN_BEFORE_PREEMPT:
                self.state=State.VEH_YEL
                self._reset(PREEMPT_CUT_TO)

        if self.state==State.VEH_GREEN:
            self._outs(True,False,False,False,True)
            if self._elapsed()>=self.tgt:
                self.state=State.VEH_YEL; self._reset(VEH_YEL_TIME)

        elif self.state==State.VEH_YEL:
            self._outs(False,True,False,False,True)
            if self._elapsed()>=self.tgt:
                self.state=State.ALL_RED_TO_PED; self._reset(ALL_RED_CLEAR)

        elif self.state==State.ALL_RED_TO_PED:
            self._outs(False,False,True,True,False)
            if self._elapsed()>=self.tgt:
                self.state=State.PED_GREEN
                extra = PED_ENTRY_BOOST if pessoa_na_faixa else 0
                self._reset(PED_GREEN_BASE + extra)
                self.last_seen_person_ts = now if pessoa_na_faixa else None
                self.tag_pending=False

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
                    self.tgt += PED_EXT_STEP

        elif self.state==State.ALL_RED_TO_VEH:
            self._outs(False,False,True,False,True)
            if self._elapsed()>=self.tgt:
                self.state=State.VEH_GREEN
                self._reset(VEH_GREEN_DEFAULT)
                self.last_seen_person_ts=None
        return self.state

    # HUD à direita
    @staticmethod
    def _put_right(frame,text,y,scale=0.9,color=(255,255,255),th=2):
        (w,_),_ = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,th)
        x = frame.shape[1]-20-w
        cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),th+1)
        cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,th)

    def overlay(self,frame,pessoa_na_faixa:bool,siren:bool,audio_db:Optional[float],peak_db:Optional[float]):
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
        self._put_right(frame,f"Sirene: {'SIM' if siren else 'NAO'}",160,0.85,(150,200,255))

        # VU meter (barra + dB)
        h, w = frame.shape[:2]
        vu_w = int(w * 0.35)
        vu_h = 18
        x1 = 20
        y1 = h - 30
        x2 = x1 + vu_w
        y2 = y1 + vu_h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60,60,60), 2)

        if audio_db is None:
            txt = "VU: N/A"
            cv2.putText(frame, txt, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        else:
            db_clamped = max(-60.0, min(0.0, audio_db))
            p = (db_clamped + 60.0) / 60.0  # 0..1
            fill = x1 + int(p * vu_w)
            cv2.rectangle(frame, (x1, y1), (fill, y2), (80,220,80), -1)

            # pico (peak hold)
            if peak_db is not None:
                peak_db_cl = max(-60.0, min(0.0, peak_db))
                pkp = (peak_db_cl + 60.0) / 60.0
                pkx = x1 + int(pkp * vu_w)
                cv2.line(frame, (pkx, y1), (pkx, y2), (0,200,255), 2)

            txt = f"VU: {audio_db:5.1f} dB"
            cv2.putText(frame, txt, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# ===================== ROI helpers =====================
def save_roi(pts,path): open(path,"w").write(json.dumps({"points":pts}))
def load_roi(path): return [tuple(p) for p in json.load(open(path))["points"]]
def in_poly(pt,poly): return cv2.pointPolygonTest(np.array(poly,np.int32),pt,False)>=0
def draw_poly(frame,poly,color,fill=True):
    pts=np.array(poly,np.int32)
    if fill:
        ov=frame.copy(); cv2.fillPoly(ov,[pts],color); frame[:]=cv2.addWeighted(ov,0.2,frame,0.8,0)
    cv2.polylines(frame,[pts],True,color,2)

# ===================== ÁUDIO (VU + Sirene) =====================
class AudioSirenDetector(threading.Thread):
    """
    Lê o áudio do RTSP com PyAV e fornece:
      - nível RMS em dBFS (audio_db)
      - pico recente (peak_db)
      - flag de sirene (siren_flag) via análise de banda 600–1800 Hz + modulação 0.5–4 Hz
    """
    def __init__(self, rtsp_url:str, sample_rate:int=16000):
        super().__init__(daemon=True)
        self.url = rtsp_url
        self.sr  = sample_rate
        self.buf = collections.deque(maxlen= self.sr*6)  # ~6s de áudio
        self._alive = True
        self._lock = threading.Lock()
        self._audio_db: Optional[float] = None
        self._peak_db: Optional[float]  = None
        self._pk_hold_until = 0.0
        self._siren_flag = False
        self._on_cnt = 0
        self._off_cnt= 0
        self.HYST_ON  = 4
        self.HYST_OFF = 6

    def stop(self): self._alive=False

    def levels(self) -> Tuple[Optional[float], Optional[float]]:
        with self._lock:
            return self._audio_db, self._peak_db

    def is_siren(self) -> bool:
        with self._lock:
            return self._siren_flag

    def run(self):
        try:
            import av
        except Exception as e:
            print("[AUDIO] PyAV não instalado:", e)
            return

        try:
            cn = av.open(self.url, timeout=5.0, options={"rtsp_transport":"udp","stimeout":"5000000"})
        except Exception as e:
            print("[AUDIO] Falha ao abrir RTSP:", e)
            return

        astreams = [s for s in cn.streams if s.type=="audio"]
        if not astreams:
            print("[AUDIO] Sem stream de ÁUDIO no RTSP.")
            return
        astream = astreams[0]

        from av.audio.resampler import AudioResampler
        resamp = AudioResampler(format="s16", layout="mono", rate=self.sr)
        last_proc = time.time()

        while self._alive:
            try:
                for pkt in cn.demux(astream):
                    if not self._alive: break
                    for frm in pkt.decode():
                        frm = resamp.resample(frm)
                        samples = frm.to_ndarray().astype(np.int16).astype(np.float32) / 32768.0
                        s = samples.flatten()
                        self.buf.extend(s)

                        # VU meter (RMS em dBFS) + peak hold (1.0s)
                        rms = float(np.sqrt(np.mean(s*s) + 1e-12))
                        db  = 20.0 * math.log10(rms + 1e-12)  # ~ -inf..0
                        now = time.time()

                        with self._lock:
                            self._audio_db = max(-60.0, min(0.0, db))
                            # peak em janela instantânea + hold 1s
                            inst_peak = float(np.max(np.abs(s)) + 1e-12)
                            pkdb = 20.0 * math.log10(inst_peak + 1e-12)
                            if self._peak_db is None or pkdb > self._peak_db or now > self._pk_hold_until:
                                self._peak_db = max(-60.0, min(0.0, pkdb))
                                self._pk_hold_until = now + 1.0

                        if now - last_proc >= 0.9:  # ~1s p/ análise de sirene
                            last_proc = now
                            self._update_siren()

            except Exception:
                time.sleep(0.05)

        try:
            cn.close()
        except: pass

    def _update_siren(self):
        # precisa de ~2s para análise
        need = int(self.sr*2.0)
        if len(self.buf) < need: return
        x = np.array(list(self.buf)[-need:], dtype=np.float32)

        # FFT por janelas
        win = 1024
        hop = 512
        n_frames = 1 + (len(x)-win)//hop if len(x)>=win else 0
        if n_frames <= 0: return

        hann = np.hanning(win).astype(np.float32)
        sr = self.sr
        band_lo, band_hi = 600.0, 1800.0

        band_energies = []
        total_energies= []
        for i in range(n_frames):
            seg = x[i*hop : i*hop+win]
            if len(seg)<win: break
            seg = seg * hann
            spec = np.fft.rfft(seg)
            mag2 = (spec.real**2 + spec.imag**2)
            freqs = np.fft.rfftfreq(win, 1.0/sr)

            band = (freqs>=band_lo) & (freqs<=band_hi)
            band_e = float(np.sum(mag2[band]) + 1e-8)
            total_e= float(np.sum(mag2) + 1e-8)

            band_energies.append(band_e)
            total_energies.append(total_e)

        band_energies = np.array(band_energies, dtype=np.float32)
        total_energies= np.array(total_energies, dtype=np.float32)
        if len(band_energies) < 3: return

        # razão banda/total e modulação na banda
        ratio = float(np.mean(band_energies / total_energies))

        be = (band_energies - np.mean(band_energies)) / (np.std(band_energies)+1e-6)
        ac = np.correlate(be, be, mode="full")
        ac = ac[len(be)-1:]
        frame_rate = sr / hop  # ~31.25 Hz
        fmin, fmax = 0.5, 4.0
        lag_min = max(1, int(frame_rate/fmax))
        lag_max = min(len(ac)-1, int(frame_rate/fmin))
        if lag_max <= lag_min: return
        peak = float(np.max(ac[lag_min:lag_max])) / (ac[0]+1e-6)

        ratio_thr = 0.05
        peak_thr  = 0.15
        positive = (ratio >= ratio_thr) and (peak >= peak_thr)

        with self._lock:
            if positive:
                self._on_cnt  = min(self.HYST_ON,  self._on_cnt + 1)
                self._off_cnt = max(0, self._off_cnt - 1)
            else:
                self._off_cnt = min(self.HYST_OFF, self._off_cnt + 1)
                self._on_cnt  = max(0, self._on_cnt - 1)

            if not self._siren_flag and self._on_cnt >= self.HYST_ON:
                self._siren_flag = True
            elif self._siren_flag and self._off_cnt >= self.HYST_OFF:
                self._siren_flag = False

# ===================== ROI / Mouse =====================
_clicked: List[Tuple[int,int]]=[]; _selecting=False
tag_sim: TagSimulator|None=None

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

    # Áudio (VU + sirene)
    audio_thr = AudioSirenDetector(CAMERA_URL)
    audio_thr.start()

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
            if cv2.waitKey(1)&0xFF==ord('q'):
                audio_thr.stop()
                return
    if roi is None:
        audio_thr.stop()
        return

    cv2.namedWindow(MAIN_WIN); cv2.setMouseCallback(MAIN_WIN,mouse_cb,param=None)

    while True:
        ok,frame=cap.read()
        if not ok: break

        draw_poly(frame,roi,C_ROI,fill=True)

        # -------- Detecção (pessoa na ROI + veículos só para HUD) --------
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
                    cv2.rectangle(frame,(x1,y1),(x2,y2),C_VEICULO,2)
                    cv2.circle(frame,(cx,cy),4,C_VEICULO,-1)
                    cv2.putText(frame,f"{label} {conf:.2f}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,C_VEICULO,2)

        # -------- TAG / teclado --------
        key=(cv2.waitKey(1)&0xFF)
        tag_sim.keypress(key)
        tag_sim.file_poll()
        pulse,_=tag_sim.poll()
        if pulse: fsm.request_tag()

        # -------- Áudio (VU + Sirene) --------
        audio_db, peak_db = audio_thr.levels()
        sirene = audio_thr.is_siren()

        # (opcional) usar sirene para preempção:
        if sirene and fsm.state==State.VEH_GREEN and fsm._elapsed()>=MIN_GREEN_BEFORE_PREEMPT:
          fsm.request_tag()

        # -------- FSM + HUD --------
        fsm.update(pessoa_na_faixa)
        fsm.overlay(frame, pessoa_na_faixa, sirene, audio_db, peak_db)
        tag_sim.draw_button(frame)

        cv2.imshow(MAIN_WIN,frame)
        if key in (ord('q'), ord('Q')):
            print("Encerrando...")
            break

    # Encerrar
    try: audio_thr.stop()
    except: pass
    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
