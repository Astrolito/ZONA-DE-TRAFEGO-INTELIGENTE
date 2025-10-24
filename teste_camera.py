from ultralytics import YOLO
import cv2
import threading
import time

# Caminho do modelo YOLOv8
model = YOLO("yolov8n.pt")

# URL da câmera Intelbras (RTSP)
# Exemplo: "rtsp://usuario:senha@192.168.100.28:554/cam/realmonitor?channel=1&subtype=0"
CAMERA_URL = "rtsp://admin:QT9RJ462@192.168.0.108:554/cam/realmonitor?channel=1&subtype=1"

# Variável global do frame
latest_frame = None
stopped = False

# Função de leitura contínua da câmera (thread)
def capture_frames():
    global latest_frame, stopped
    cap = cv2.VideoCapture(CAMERA_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduz buffer interno

    if not cap.isOpened():
        print("❌ Não foi possível abrir a câmera. Verifique o IP, usuário e senha.")
        stopped = True
        return

    while not stopped:
        ret, frame = cap.read()
        if ret:
            latest_frame = frame
        else:
            print("⚠️ Falha ao capturar frame.")
            time.sleep(0.1)

    cap.release()

# Inicia thread de captura
thread = threading.Thread(target=capture_frames, daemon=True)
thread.start()

# Espera até que o primeiro frame chegue
while latest_frame is None and not stopped:
    time.sleep(0.1)

print("✅ Captura iniciada com sucesso!")

# Loop principal
while not stopped:
    if latest_frame is None:
        continue

    frame = latest_frame.copy()

    # Faz detecção
    results = model(frame, stream=True)

    # Exibe resultados
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            if conf > 0.5:
                (x1, y1, x2, y2) = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Detecção - YOLOv8 (Intelbras)", frame)

    if cv2.waitKey(1) == ord('q'):
        stopped = True
        break

cv2.destroyAllWindows()
print("🛑 Encerrado com segurança.")