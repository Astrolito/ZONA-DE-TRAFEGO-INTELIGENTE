"""
esp_comm.py — Camada de comunicação serial PC ⇄ ESP32 para o projeto do semáforo

Protocolo (linhas ASCII terminadas em '\n'):
    PC → ESP:
        OUT,<b0..b4>     # bits na ordem: veh_g,veh_y,veh_r,ped_w,ped_dw (ex.: OUT,10110)
        PING             # ESP deve responder 'PONG'
    ESP → PC:
        TAG,<uidhex>     # UID lida no RC522 (ex.: TAG,DEADBEEF)
        BTN,PED          # botão físico pressionado
        HB               # heartbeat (1s) opcional
        PONG             # resposta ao PING

Uso típico no loop da CÂMERA:
    from esp_comm import EspLink, outputs_to_bits

    link = EspLink(port="COM5")  # ou None para auto-detecção
    link.start()

    while True:
        # ... leia frame da câmera, compute 'outs' (dict) ...
        bits = outputs_to_bits(outs)
        link.send_outputs(bits)            # envia OUT quando quiser

        for ev in link.poll_events():      # consome eventos do ESP
            if ev["type"] == "TAG":
                # tratar pré-empção etc.
                pass
            elif ev["type"] == "BTN":
                pass
        # ...
    link.stop()
"""

from __future__ import annotations
import os
import threading
import queue
from dataclasses import dataclass
from typing import Optional, List, Dict

import serial
import serial.tools.list_ports


# ===================== Utilitário para montar bits de saída =====================
def outputs_to_bits(outs: Dict[str, bool]) -> str:
    """
    Converte o dicionário de saídas para a string de bits esperada pelo ESP.
    Ordem: veh_g, veh_y, veh_r, ped_w, ped_dw
    """
    return "".join([
        '1' if outs.get("veh_g") else '0',
        '1' if outs.get("veh_y") else '0',
        '1' if outs.get("veh_r") else '0',
        '1' if outs.get("ped_w") else '0',
        '1' if outs.get("ped_dw") else '0',
    ])


# ===================== Evento vindo do ESP =====================
@dataclass
class EspEvent:
    type: str                 # "TAG" | "BTN" | "HB" | "PONG" | "RAW"
    payload: Optional[str]    # UID em hex para TAG, None p/ BTN/HB/PONG, ou linha crua para RAW


# ===================== Camada de link serial =====================
class EspLink:
    def __init__(self, port: Optional[str] = None, baud: int = 115200, timeout_s: float = 0.01):
        """
        port:
            - "COM5" (Windows), "/dev/ttyUSB0" ou "/dev/ttyACM0" (Linux), "/dev/cu.SLAB_USBtoUART" (macOS)
            - None → tentativa de auto-detecção
        """
        self.port = port
        self.baud = baud
        self.timeout_s = timeout_s

        self._ser: Optional[serial.Serial] = None
        self._rx_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._events_q: "queue.Queue[EspEvent]" = queue.Queue(maxsize=256)

        # cache para OUT: evita floodar a serial com linhas idênticas
        self._last_bits: Optional[str] = None

    # ---------- API pública ----------
    def start(self) -> bool:
        """Abre a porta e inicia a thread de recepção. Retorna True se conectou."""
        if self._ser and self._ser.is_open:
            return True

        port = self.port or self._autodetect_port()
        if not port:
            print("[ESP] Porta não encontrada (conecte o ESP32 ou informe --port).")
            return False

        try:
            self._ser = serial.Serial(port, self.baud, timeout=self.timeout_s)
            self._running.set()
            self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
            self._rx_thread.start()
            print(f"[ESP] Conectado em {port}")
            return True
        except Exception as e:
            print(f"[ESP] Falha ao abrir {port}: {e}")
            self._ser = None
            return False

    def stop(self) -> None:
        """Encerra thread e fecha a serial."""
        self._running.clear()
        if self._rx_thread and self._rx_thread.is_alive():
            self._rx_thread.join(timeout=1.0)
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None

    def send_outputs(self, bits: str, force: bool = False) -> None:
        """
        Envia OUT,<bits> para o ESP (ex.: '10110').
        Usa cache para não repetir a mesma linha; use force=True para reenviar mesmo sem mudança.
        """
        if not self._is_open():
            return
        bits = (bits or "").strip()
        if not bits or (not force and bits == self._last_bits):
            return
        self._write_line(f"OUT,{bits}")
        self._last_bits = bits

    def send_ping(self) -> None:
        """Envia PING; ESP deve responder 'PONG'."""
        if self._is_open():
            self._write_line("PING")

    def poll_events(self) -> List[Dict[str, Optional[str]]]:
        """
        Retorna uma lista de eventos pendentes (não bloqueante).
        Formato de cada item: {"type": "TAG"|"BTN"|"HB"|"PONG"|"RAW", "payload": str|None}
        """
        out: List[Dict[str, Optional[str]]] = []
        try:
            while True:
                ev = self._events_q.get_nowait()
                out.append({"type": ev.type, "payload": ev.payload})
        except queue.Empty:
            pass
        return out

    # ---------- Internos ----------
    def _is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def _write_line(self, line: str) -> None:
        try:
            self._ser.write((line.strip() + "\n").encode("utf-8"))
        except Exception as e:
            print("[ESP] Erro ao enviar:", e)

    def _rx_loop(self) -> None:
        buf = bytearray()
        while self._running.is_set() and self._is_open():
            try:
                b = self._ser.read(1)
                if not b:
                    continue
                if b in (b"\n", b"\r"):
                    if buf:
                        line = buf.decode(errors="ignore").strip()
                        self._handle_line(line)
                        buf.clear()
                else:
                    buf.extend(b)
                    if len(buf) > 512:  # proteção contra linhas malformadas gigantes
                        buf.clear()
            except Exception:
                break

    def _handle_line(self, line: str) -> None:
        # Parse simples baseado no protocolo definido
        line_up = line.upper()
        if line_up.startswith("TAG,"):
            uid = line.split(",", 1)[1].strip()
            self._push_event(EspEvent("TAG", uid))
        elif line_up == "BTN,PED":
            self._push_event(EspEvent("BTN", None))
        elif line_up == "HB":
            self._push_event(EspEvent("HB", None))
        elif line_up == "PONG":
            self._push_event(EspEvent("PONG", None))
        else:
            # Linha desconhecida — ainda entregamos para debug
            self._push_event(EspEvent("RAW", line))

    def _push_event(self, ev: EspEvent) -> None:
        try:
            self._events_q.put_nowait(ev)
        except queue.Full:
            # fila cheia — descartamos o mais antigo para priorizar o evento atual
            try:
                _ = self._events_q.get_nowait()
                self._events_q.put_nowait(ev)
            except Exception:
                pass

    @staticmethod
    def _autodetect_port() -> Optional[str]:
        """
        Tenta encontrar uma porta típica de conversores USB-Serial (CP210x, CH340, FTDI, SiLabs).
        """
        for p in serial.tools.list_ports.comports():
            name = f"{p.device} {p.description}".upper()
            if any(k in name for k in ["USB", "CP210", "CH340", "FTDI", "SILABS", "CP210X", "UART"]):
                return p.device
        # fallback comum
        if os.name == "nt":
            return "COM5"
        return "/dev/ttyUSB0"


# ===================== Exemplo rápido de uso isolado =====================
if __name__ == "__main__":
    # Demonstração minimalista: conecta, manda PING a cada 2s e imprime eventos recebidos.
    import time

    link = EspLink(port=None)  # None = auto
    if not link.start():
        raise SystemExit(1)

    try:
        t0 = time.monotonic()
        while True:
            # Envia um PING a cada ~2s
            if (time.monotonic() - t0) >= 2.0:
                link.send_ping()
                t0 = time.monotonic()

            # Lê e exibe eventos
            for ev in link.poll_events():
                print("[ESP EVENT]", ev)

            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        link.stop()
