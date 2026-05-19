import cv2
import time
import argparse
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


def run_yolo_deepsort(video_path, output_path, model_name="yolov8n.pt", conf_thres=0.4):
    model = YOLO(model_name)

    tracker = DeepSort(
        max_age=30,
        n_init=3,
        max_cosine_distance=0.4
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        input_fps,
        (width, height)
    )

    frame_count = 0
    total_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        results = model(frame, verbose=False)[0]

        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if conf < conf_thres:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w = x2 - x1
            h = y2 - y1

            detections.append(([x1, y1, w, h], conf, cls_id))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()

            cls_id = track.det_class
            class_name = COCO_CLASSES[cls_id] if cls_id is not None and cls_id < len(COCO_CLASSES) else "object"

            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id} | {class_name}",
                (int(l), int(t) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        elapsed = time.time() - start_time
        total_time += elapsed
        frame_count += 1

        fps = 1 / elapsed if elapsed > 0 else 0

        cv2.putText(
            frame,
            f"YOLOv8 + DeepSORT | FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        writer.write(frame)

    cap.release()
    writer.release()

    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"YOLOv8 + DeepSORT finished.")
    print(f"Frames: {frame_count}")
    print(f"Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="outputs/yolo_deepsort_output.mp4")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()

    run_yolo_deepsort(args.video, args.output, args.model, args.conf)
