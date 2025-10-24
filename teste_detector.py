from ultralytics import YOLO
import cv2

# Caminho do modelo YOLOv8 (você pode usar 'yolov8n.pt' para leve)
model = YOLO("yolov8n.pt")

# URL da sua câmera Intelbras IM4C
CAMERA_URL = 0 #"rtsp://admin:Naoesquecer1@192.168.100.28:554/cam/realmonitor?channel=1&subtype=0"

# Abre o vídeo
cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():      
    print("❌ Não foi possível abrir a câmera. Verifique o IP, usuário e senha.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Falha ao capturar imagem.")
        break

    # Faz detecção
    results = model(frame, stream=True)

    # Exibe resultados
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            if conf > 0.5:
                (x1, y1, x2, y2) = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Detecção - YOLOv8", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
