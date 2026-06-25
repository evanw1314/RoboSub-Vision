import time

import cv2
import numpy as np
from ultralytics import YOLO

from color_correction import (
    add_label,
    apply_underwater_effect,
    correct_underwater_image,
)

MODEL_SIZE = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.4
WINDOW_NAME = "YOLOv8 Testing"

DEFAULT_CAMERA_IDX = 0
FUDGE_FACTOR = 1e-6
REFRESH_TIME = 1

FPS_FONT_SCALE = 0.6
FPS_COLOR = (0, 255, 120)
FPS_THICKNESS = 2
FPS_X_OFFSET = 115
FPS_Y_OFFSET = 12

DETECTIONS_COORDS = (8, 50)
DETECTIONS_FONT_SCALE = 0.5
DETECTIONS_COLOR = (0, 255, 120)
DETECTIONS_THICKNESS = 1

TRACKBAR_STARTPOS = 50
TRACKBAR_MAXPOS = 100

PAD_COLOR = (0, 0, 0)

print(f"Loading {MODEL_SIZE}.")
model = YOLO(MODEL_SIZE)
print("Model ready.")


def run_yolo(frame):
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    annotated = results[0].plot()

    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        detections.append(
            {
                "label": model.names[class_id],
                "confidence": float(box.conf[0]),
                "box": tuple(map(int, box.xyxy[0])),
            }
        )

    return annotated, detections


def draw_hud(frame, detections, fps):
    height, width = frame.shape[:2]

    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (width - FPS_X_OFFSET, height - FPS_Y_OFFSET),
        cv2.FONT_HERSHEY_SIMPLEX,
        FPS_FONT_SCALE,
        FPS_COLOR,
        FPS_THICKNESS,
    )

    if detections:
        summary_parts = [
            f"{d['label']} ({int(d['confidence'] * 100)}%)" for d in detections
        ]
        summary = ", ".join(summary_parts)
    else:
        summary = "Nothing detected"

    cv2.putText(
        frame,
        summary,
        DETECTIONS_COORDS,
        cv2.FONT_HERSHEY_SIMPLEX,
        DETECTIONS_FONT_SCALE,
        DETECTIONS_COLOR,
        DETECTIONS_THICKNESS,
    )

    return frame


def main():
    cap = cv2.VideoCapture(DEFAULT_CAMERA_IDX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(
        "Depth %", WINDOW_NAME, TRACKBAR_STARTPOS, TRACKBAR_MAXPOS, lambda x: None
    )

    prev_time = time.time()

    print(f"{WINDOW_NAME} running!")
    print("Left: Original | Center: Degraded | Right: Corrected with YOLOv8")
    print("Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        depth = cv2.getTrackbarPos("Depth %", WINDOW_NAME) / 100.0
        width = frame.shape[1]

        degraded = apply_underwater_effect(frame, depth)
        corrected = correct_underwater_image(degraded)
        annotated, detections = run_yolo(corrected)

        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, FUDGE_FACTOR)
        prev_time = curr_time

        panel_original = add_label(frame, "Original", width)
        panel_degraded = add_label(
            degraded, f"Degraded  (Depth: {int(depth * 100)}%)", width
        )

        annotated = draw_hud(annotated, detections, fps)
        panel_detected = add_label(annotated, "Corrected + YOLOv8", width)

        comparison = np.hstack([panel_original, panel_degraded, panel_detected])

        win_rect = cv2.getWindowImageRect(WINDOW_NAME)
        win_w, win_h = win_rect[2], win_rect[3]

        if win_w > 0 and win_h > 0:
            img_h, img_w = comparison.shape[:2]
            scale = win_w / img_w
            new_w = win_w
            new_h = int(img_h * scale)
            comparison = cv2.resize(
                comparison, (new_w, new_h), interpolation=cv2.INTER_AREA
            )

            if new_h < win_h:
                pad_top = (win_h - new_h) // 2
                pad_bot = win_h - new_h - pad_top
                comparison = cv2.copyMakeBorder(
                    comparison,
                    pad_top,
                    pad_bot,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=PAD_COLOR,
                )

        cv2.imshow(WINDOW_NAME, comparison)

        if cv2.waitKey(REFRESH_TIME) & 0xFF == ord("q"):
            print("Q pressed — shutting down.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
