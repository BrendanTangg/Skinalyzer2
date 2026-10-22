from ultralytics import YOLO
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os

from skin_redness import RednessAnalyzer
from skin_pigmentation import PigmentationAnalyzer
from skin_oiliness import OilinessAnalyzer

model_path = 'models/face_landmarker.task'
acne_model = YOLO("skinalyzer2.pt")
redness = RednessAnalyzer(threshold=8.0, smoothing=0.15)
pigmentation = PigmentationAnalyzer(darkness=5.0, smoothing=0.15)
oiliness = OilinessAnalyzer(brightness=15.0, sat_drop=0.75, smoothing=0.15)
show_heatmap = False
show_spots = True
show_shine = True

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_face_result = None

# Create a face landmarker instance with the live stream mode:
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # Printing the whole result every frame floods the console and slows the loop
    global latest_face_result
    latest_face_result = result


def score_color(score):
    """Green -> yellow -> red (BGR) as the redness score rises."""
    if score < 20:
        return (0, 200, 0)
    if score < 40:
        return (0, 200, 255)
    return (0, 0, 255)


def draw_label(img, text, x, y, color):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    result_callback=print_result)

with FaceLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
    stream = cv2.VideoCapture(0)

    if not stream.isOpened():
        print("No stream :(")
        exit()

    start_time = time.monotonic()
    last_timestamp_ms = -1

    while (True):
        ret, frame = stream.read()
        if not ret:
            print ("No more stream :(")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.monotonic()-start_time)*1000)

        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1

        last_timestamp_ms = timestamp_ms
        landmarker.detect_async(mp_image, timestamp_ms)

        # Run acne detection on the current webcam frame
        acne_results = acne_model.predict(
            source=frame,
            conf=0.2,
            verbose=False
        )
        acne_boxes = acne_results[0].boxes.xyxy.cpu().numpy()

        # Keep the clean frame for color analysis. The plotted frame has
        # colored boxes and text drawn on it, which would corrupt redness.
        raw_frame = frame

        # Draw YOLO acne bounding boxes onto the frame
        frame = acne_results[0].plot()

        if latest_face_result and latest_face_result.face_landmarks:
            h, w = frame.shape[:2]

            face_landmarks = latest_face_result.face_landmarks[0]

            x_coords = []
            y_coords = []

            for lm in face_landmarks:
                x_coords.append(int(lm.x * w))
                y_coords.append(int(lm.y * h))

            # Calculate face dimensions
            face_width = max(x_coords) - min(x_coords)
            face_height = max(y_coords) - min(y_coords)

            # MediaPipe landmarks for approximate cheek locations
            left_cheek_lm = face_landmarks[50]
            right_cheek_lm = face_landmarks[280]

            # MediaPipe landmark near the center of the forehead
            forehead_lm = face_landmarks[10]

            # Convert normalized coordinates to pixel coordinates
            left_cheek_x = int(left_cheek_lm.x * w)
            left_cheek_y = int(left_cheek_lm.y * h)

            right_cheek_x = int(right_cheek_lm.x * w)
            right_cheek_y = int(right_cheek_lm.y * h)

            # Convert forehead landmark to pixel coordinates
            forehead_x = int(forehead_lm.x * w)
            forehead_y = int(forehead_lm.y * h)

            cheek_y_offset = int(face_height * 0.07)

            left_cheek_y += cheek_y_offset
            right_cheek_y += cheek_y_offset

            # Move forehead region slightly downward
            forehead_y_offset = int(face_height * 0.04)
            forehead_y += forehead_y_offset

            cheek_width = int(face_width * 0.12)
            cheek_height = int(face_height * 0.15)

            # Forehead region size
            forehead_width = int(face_width * 0.32)
            forehead_height = int(face_height * 0.08)

            # Left cheek boundaries
            left_x1 = max(0, left_cheek_x - cheek_width)
            left_x2 = min(w, left_cheek_x + cheek_width)
            left_y1 = max(0, left_cheek_y - cheek_height)
            left_y2 = min(h, left_cheek_y + cheek_height)

            # Right cheek boundaries
            right_x1 = max(0, right_cheek_x - cheek_width)
            right_x2 = min(w, right_cheek_x + cheek_width)
            right_y1 = max(0, right_cheek_y - cheek_height)
            right_y2 = min(h, right_cheek_y + cheek_height)

            # Forehead boundaries
            forehead_x1 = max(0, forehead_x - forehead_width)
            forehead_x2 = min(w, forehead_x + forehead_width)
            forehead_y1 = max(0, forehead_y - forehead_height)
            forehead_y2 = min(h, forehead_y + forehead_height)

            regions = {
                "left_cheek": (left_x1, left_y1, left_x2, left_y2),
                "right_cheek": (right_x1, right_y1, right_x2, right_y2),
                "forehead": (forehead_x1, forehead_y1, forehead_x2, forehead_y2),
            }

            # ---- Redness ----
            analysis = redness.analyze(raw_frame, face_landmarks, regions, acne_boxes)
            if analysis is not None:
                red_results, excess, skin = analysis

                if show_heatmap:
                    redness.draw_heatmap(frame, excess, skin)

                for name, (x1, y1, x2, y2) in regions.items():
                    if name in red_results:
                        s = red_results[name]["score"]
                        draw_label(frame, f"red {s:.0f}", x1, y1 - 6, score_color(s))

                face = red_results.get("face")
                if face:
                    draw_label(frame,
                               f"Redness: {face['score']:.0f}  "
                               f"({face['red_pct']:.1f}% area)  "
                               f"baseline a*: {face['baseline_a']:.1f}",
                               10, 25, score_color(face["score"]))

            # ---- Pigmentation ----
            pig = pigmentation.analyze(raw_frame, face_landmarks, regions, acne_boxes)
            if pig is not None:
                pig_results, spot_mask = pig

                if show_spots:
                    pigmentation.draw_spots(frame, spot_mask)

                for name, (x1, y1, x2, y2) in regions.items():
                    if name in pig_results:
                        s = pig_results[name]["score"]
                        draw_label(frame, f"pig {s:.0f}", x1, y2 + 15, score_color(s))

                face = pig_results.get("face")
                if face:
                    draw_label(frame,
                               f"Pigmentation: {face['score']:.0f}  "
                               f"spots: {face['spots']:.0f}  "
                               f"({face['spot_pct']:.1f}% area)  "
                               f"uneven: {face['uneven']:.2f}",
                               10, 45, score_color(face["score"]))

            # ---- Oiliness ----
            oil = oiliness.analyze(raw_frame, face_landmarks, regions, acne_boxes)
            if oil is not None:
                oil_results, shine_mask = oil

                if show_shine:
                    oiliness.draw_shine(frame, shine_mask)

                for name, (x1, y1, x2, y2) in regions.items():
                    if name in oil_results:
                        s = oil_results[name]["score"]
                        draw_label(frame, f"oil {s:.0f}", x1, y2 + 30, score_color(s))

                face = oil_results.get("face")
                tz = oil_results.get("t_zone")
                if face:
                    tz_text = f"  T-zone: {tz['score']:.0f}" if tz else ""
                    draw_label(frame,
                               f"Oiliness: {face['score']:.0f}  "
                               f"({face['shine_pct']:.1f}% shiny){tz_text}",
                               10, 65, score_color(face["score"]))

            cv2.rectangle(
                frame,
                (left_x1, left_y1),
                (left_x2, left_y2),
                (255, 0, 0),
                2
            )

            cv2.rectangle(
                frame,
                (right_x1, right_y1),
                (right_x2, right_y2),
                (255, 0, 0),
                2
            )

            cv2.rectangle(
                frame,
                (forehead_x1, forehead_y1),
                (forehead_x2, forehead_y2),
                (255, 0, 0),
                2
            )

        draw_label(frame, "h: heatmap  p: spots  o: shine  q: quit", 10, frame.shape[0] - 10, (255, 255, 255))
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('h'):
            show_heatmap = not show_heatmap
        if key == ord('p'):
            show_spots = not show_spots
        if key == ord('o'):
            show_shine = not show_shine

    stream.release()
cv2.destroyAllWindows()
