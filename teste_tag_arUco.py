import cv2

cap = cv2.VideoCapture(0)

# ArUco v5+
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

TARGET_ID = 17

while True:
    ok, frame = cap.read()
    if not ok:
        break

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        for i, cid in enumerate(ids.flatten()):
            pts = corners[i][0].astype(int)
            cx = int(pts[:,0].mean())
            cy = int(pts[:,1].mean())
            
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.circle(frame, (cx,cy), 8, (0,0,255), -1)
            cv2.putText(frame, f"ID {cid}", (cx-30, cy-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            if cid == TARGET_ID:
                cv2.putText(frame, "EMERGENCIA DETECTADA",
                            (20,50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0,255,0), 3)

    cv2.imshow("Teste ArUco - OpenCV 5+", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
