from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

# YOLOv8 modelini yükle
model = YOLO("yolov8n.pt")  # küçük ve hızlı model, istersen "yolov8s.pt" gibi varyantlara geçebilirsin

# DeepSORT başlat
tracker = DeepSort(max_age=30)

# Webcam başlat
cap = cv2.VideoCapture(0)  # 0 = varsayılan kamera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO ile insan tespiti
    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:  # sadece insanları takip et (class 0 = person)
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # DeepSORT ile takip
    tracks = tracker.update_tracks(detections, frame=frame)

    # Takip kutularını çiz
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow("YOLOv8 + DeepSORT Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
