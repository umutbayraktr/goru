from ultralytics import YOLO
import cv2
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
from tqdm import tqdm

# 🔁 Yalnızca bu satırı düzenle
video_path = r"C:\Users\user\Desktop\goru\test.mp4"  # Video yolunu buraya yaz
output_path = r"C:\Users\user\Desktop\goru\cikti.mp4"  # İstersen çıktı yolunu yaz, yoksa None yap

def process_video(video_path, output_path=None, show_video=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("yolov8n.pt").to(device)

    tracker = DeepSort(max_age=30)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Video açılamadı: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    vehicle_classes = ['bicycle', 'motorcycle', 'car', 'van', 'bus', 'truck', 'trailer']
    colors = {
        'bicycle': (0, 255, 0),
        'motorcycle': (0, 255, 255),
        'car': (0, 0, 255),
        'van': (255, 0, 0),
        'bus': (255, 0, 255),
        'truck': (255, 255, 0),
        'trailer': (128, 0, 128),
        'unknown': (192, 192, 192),
    }

    frame_count = 0
    id_to_class = {}
    tracked_ids = set()
    print(f"Video işleniyor: {video_path}")
    pbar = tqdm(total=total_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        pbar.update(1)
        results = model(frame)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id] if cls_id < len(model.names) else "unknown"
            if class_name in vehicle_classes:
                bbox = box.xyxy.squeeze().cpu().numpy().astype(int)
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, class_name))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            tracked_ids.add(track_id)
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            if track_id not in id_to_class and len(detections) > 0:
                for det in detections:
                    _, _, det_class = det
                    id_to_class[track_id] = det_class

            class_name = id_to_class.get(track_id, 'unknown')
            color = colors.get(class_name, (192, 192, 192))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} ID:{track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if show_video:
            cv2.imshow("Araç Takip", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if output_path:
            out.write(frame)

    pbar.close()
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

    print(f"\nTamamlandı! Toplam kare: {frame_count}")
    print(f"Takip edilen ID sayısı: {len(tracked_ids)}")
    if output_path:
        print(f"Sonuç kaydedildi: {output_path}")


if __name__ == "__main__":
    process_video(video_path=video_path, output_path=output_path, show_video=True)
