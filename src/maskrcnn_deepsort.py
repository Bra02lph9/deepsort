import cv2
import time
import argparse
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from deep_sort_realtime.deepsort_tracker import DeepSort


COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
    "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
    "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


def run_maskrcnn_deepsort(video_path, output_path, conf_thres=0.6, mask_thres=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()

    transform = T.Compose([T.ToTensor()])

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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(rgb).to(device)

        with torch.no_grad():
            prediction = model([img_tensor])[0]

        detections = []
        instance_masks = []

        boxes = prediction["boxes"].detach().cpu().numpy()
        scores = prediction["scores"].detach().cpu().numpy()
        labels = prediction["labels"].detach().cpu().numpy()
        masks = prediction["masks"].detach().cpu().numpy()

        for box, score, label, mask in zip(boxes, scores, labels, masks):
            if score < conf_thres:
                continue

            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            binary_mask = mask[0] > mask_thres

            detections.append(([x1, y1, w, h], float(score), int(label)))
            instance_masks.append(binary_mask)

        tracks = tracker.update_tracks(
            detections,
            frame=frame,
            instance_masks=instance_masks if len(instance_masks) > 0 else None
        )

        overlay = frame.copy()

        for mask in instance_masks:
            overlay[mask] = (0, 255, 255)

        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()

            cls_id = track.det_class
            class_name = COCO_CLASSES[cls_id] if cls_id is not None and cls_id < len(COCO_CLASSES) else "object"

            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id} | {class_name}",
                (int(l), int(t) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        elapsed = time.time() - start_time
        total_time += elapsed
        frame_count += 1

        fps = 1 / elapsed if elapsed > 0 else 0

        cv2.putText(
            frame,
            f"Mask R-CNN + DeepSORT | FPS: {fps:.2f}",
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
    print(f"Mask R-CNN + DeepSORT finished.")
    print(f"Frames: {frame_count}")
    print(f"Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="outputs/maskrcnn_deepsort_output.mp4")
    parser.add_argument("--conf", type=float, default=0.6)
    args = parser.parse_args()

    run_maskrcnn_deepsort(args.video, args.output, args.conf)
