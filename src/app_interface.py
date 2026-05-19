import cv2
import time
import threading
import torch
import numpy as np
import customtkinter as ctk

from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import torchvision.transforms as T
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights
)


class VideoDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 / DeepSORT / Mask R-CNN Interface")
        self.root.geometry("1150x760")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.video_path = None
        self.cap = None
        self.running = False

        self.yolo_model = None
        self.maskrcnn_model = None
        self.tracker = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = T.Compose([T.ToTensor()])

        self.mode_var = ctk.StringVar(value="YOLOv8 Detection")

        self.title = ctk.CTkLabel(
            root,
            text="Object Detection & Tracking Interface",
            font=("Arial", 26, "bold")
        )
        self.title.pack(pady=15)

        self.video_label = ctk.CTkLabel(root, text="")
        self.video_label.pack(pady=10)

        self.control_frame = ctk.CTkFrame(root)
        self.control_frame.pack(pady=10)

        self.mode_menu = ctk.CTkOptionMenu(
            self.control_frame,
            values=[
                "YOLOv8 Detection",
                "YOLOv8 + DeepSORT",
                "Mask R-CNN + DeepSORT"
            ],
            variable=self.mode_var
        )
        self.mode_menu.grid(row=0, column=0, padx=10, pady=10)

        self.btn_open = ctk.CTkButton(
            self.control_frame,
            text="Choisir vidéo",
            command=self.open_video
        )
        self.btn_open.grid(row=0, column=1, padx=10)

        self.btn_start = ctk.CTkButton(
            self.control_frame,
            text="Démarrer",
            command=self.start_detection
        )
        self.btn_start.grid(row=0, column=2, padx=10)

        self.btn_stop = ctk.CTkButton(
            self.control_frame,
            text="Stop",
            fg_color="red",
            command=self.stop_detection
        )
        self.btn_stop.grid(row=0, column=3, padx=10)

        self.status = ctk.CTkLabel(
            root,
            text=f"Device utilisé : {self.device.upper()} | Aucune vidéo sélectionnée",
            font=("Arial", 15)
        )
        self.status.pack(pady=10)

    def open_video(self):
        self.video_path = filedialog.askopenfilename(
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )

        if self.video_path:
            self.status.configure(text=f"Vidéo sélectionnée : {self.video_path}")

    def load_yolo(self):
        if self.yolo_model is None:
            self.status.configure(text="Chargement YOLOv8...")
            self.yolo_model = YOLO("yolov8n.pt")

    def load_maskrcnn(self):
        if self.maskrcnn_model is None:
            self.status.configure(text="Chargement Mask R-CNN...")
            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            self.maskrcnn_model = maskrcnn_resnet50_fpn(weights=weights)
            self.maskrcnn_model.to(self.device)
            self.maskrcnn_model.eval()

    def reset_tracker(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.4
        )

    def start_detection(self):
        if not self.video_path:
            self.status.configure(text="Choisis d'abord une vidéo.")
            return

        if self.running:
            return

        mode = self.mode_var.get()

        if mode in ["YOLOv8 Detection", "YOLOv8 + DeepSORT"]:
            self.load_yolo()

        if mode == "Mask R-CNN + DeepSORT":
            self.load_maskrcnn()

        if "DeepSORT" in mode:
            self.reset_tracker()

        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            self.status.configure(text="Erreur : impossible d'ouvrir la vidéo.")
            return

        self.running = True

        thread = threading.Thread(target=self.detect_video)
        thread.daemon = True
        thread.start()

    def detect_video(self):
        mode = self.mode_var.get()

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                break

            start_time = time.time()

            if mode == "YOLOv8 Detection":
                frame = self.process_yolo_detection(frame)

            elif mode == "YOLOv8 + DeepSORT":
                frame = self.process_yolo_deepsort(frame)

            elif mode == "Mask R-CNN + DeepSORT":
                frame = self.process_maskrcnn_deepsort(frame)

            fps = 1 / (time.time() - start_time + 1e-6)

            cv2.putText(
                frame,
                f"{mode} | FPS: {fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            self.show_frame(frame)

        self.stop_detection()

    def process_yolo_detection(self, frame):
        results = self.yolo_model(frame, conf=0.4, verbose=False)
        return results[0].plot()

    def process_yolo_deepsort(self, frame):
        results = self.yolo_model(frame, conf=0.4, verbose=False)[0]

        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w = x2 - x1
            h = y2 - y1

            detections.append(([x1, y1, w, h], conf, cls_id))

        tracks = self.tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()

            cls_id = track.det_class
            class_name = self.yolo_model.names[cls_id] if cls_id is not None else "object"

            cv2.rectangle(
                frame,
                (int(l), int(t)),
                (int(r), int(b)),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID {track_id} | {class_name}",
                (int(l), int(t) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return frame

    def process_maskrcnn_deepsort(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(rgb).to(self.device)

        with torch.no_grad():
            prediction = self.maskrcnn_model([img_tensor])[0]

        detections = []
        masks_list = []

        boxes = prediction["boxes"].detach().cpu().numpy()
        scores = prediction["scores"].detach().cpu().numpy()
        labels = prediction["labels"].detach().cpu().numpy()
        masks = prediction["masks"].detach().cpu().numpy()

        for box, score, label, mask in zip(boxes, scores, labels, masks):
            if score < 0.6:
                continue

            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            binary_mask = mask[0] > 0.5

            detections.append(([x1, y1, w, h], float(score), int(label)))
            masks_list.append(binary_mask)

        tracks = self.tracker.update_tracks(
            detections,
            frame=frame,
            instance_masks=masks_list if len(masks_list) > 0 else None
        )

        overlay = frame.copy()

        for mask in masks_list:
            overlay[mask] = (0, 255, 255)

        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()

            cv2.rectangle(
                frame,
                (int(l), int(t)),
                (int(r), int(b)),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID {track_id}",
                (int(l), int(t) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        return frame

    def show_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1000, 560))

        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)

        self.video_label.configure(image=photo)
        self.video_label.image = photo

    def stop_detection(self):
        self.running = False

        if self.cap:
            self.cap.release()
            self.cap = None

        self.status.configure(text="Détection arrêtée.")


if __name__ == "__main__":
    root = ctk.CTk()
    app = VideoDetectionApp(root)
    root.mainloop()
