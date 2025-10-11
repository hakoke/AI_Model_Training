"""PPE detection demo using a pretrained YOLOv8 model.

Run in three modes:
  - image: process a single image file
  - video: process a video file (frame-by-frame)
  - webcam: run live inference from a webcam device

The script annotates detections, displays them (unless --show False),
and saves the results to the output/ directory automatically.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    FaceAnalysis = None


DEFAULT_MODEL_NAME = "best_all6.pt"
MODEL_DIR = Path("models")
DEFAULT_MODEL_PATH = MODEL_DIR / DEFAULT_MODEL_NAME
OUTPUT_DIR = Path("output")
INPUT_DIR = Path("input")
IMAGES_DIR = INPUT_DIR / "images"
VIDEOS_DIR = INPUT_DIR / "videos"
EMPLOYEE_DB_DIR = Path("employee_db")

# No label overrides are needed now that the unified checkpoint uses the
# standardized 15-class schema directly.
LABEL_OVERRIDES: dict[int, str] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PPE detection with YOLOv8 on images, videos, or webcam.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["image", "video", "webcam"],
        help="Select the type of source to run inference on.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help=(
            "Path to the input file or webcam index. "
            "For webcam mode you can omit this argument to use index 0."
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for filtering detections.",
    )
    parser.add_argument(
        "--show",
        default="True",
        choices=["True", "False"],
        help="Show a GUI window with live detections.",
    )
    parser.add_argument(
        "--youtube-url",
        default=None,
        help="Download a YouTube video directly into input/videos and run detection (video mode only).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed detection information for troubleshooting.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help=(
            "Path or Ultralytics HUB identifier for the YOLOv8 model. "
            "Defaults to the Ultralytics PPE checkpoint (ppe.pt)."
        ),
    )
    parser.add_argument(
        "--person-model",
        default="yolov8n.pt",
        help=(
            "Secondary model identifier or path for person detection. "
            "Defaults to yolov8n.pt, which downloads automatically if missing."
        ),
    )
    parser.add_argument(
        "--enable-face-recognition",
        action="store_true",
        help="Enable ArcFace face recognition for person identification. Requires employee_db/ with face images.",
    )

    args = parser.parse_args()
    args.show = args.show.lower() == "true"
    return args


def validate_source(mode: str, source: Optional[str]) -> str:
    if mode in {"image", "video"}:
        if not source:
            print("[ERROR] --source is required for image and video modes.")
            sys.exit(1)
        path = Path(source)
        if not path.exists():
            print(f"[ERROR] Source path not found: {path}")
            sys.exit(1)
        return str(path)

   
    if source is None:
        return "0"

    return source


def timestamped_name(prefix: str, suffix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{suffix}"


def resolve_model_spec(model_arg: str) -> str:
    path = Path(model_arg)
    if path.exists():
        print(f"[INFO] Using local model weights: {path}")
        return str(path)

    if model_arg == str(DEFAULT_MODEL_PATH):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if DEFAULT_MODEL_PATH.exists():
            print(f"[INFO] Using cached default model weights: {DEFAULT_MODEL_PATH}")
            return str(DEFAULT_MODEL_PATH)

        print(
            "[ERROR] Default PPE weights not found. Please download a PPE-trained YOLOv8 checkpoint and "
            "place it at models/keremberke_yolov8m_ppe.pt, or provide a custom path via --model."
        )
        print("[HINT] Example: python main.py --mode image --source ... --model yolov8n.pt")
        print("[HINT] Ultralytics or any other PPE checkpoints can be placed in models/.")
        sys.exit(1)

    print(f"[INFO] Using Ultralytics model identifier: {model_arg}")
    return model_arg


def load_model(model_spec: str, *, apply_overrides: bool = False) -> YOLO:
    print(f"[INFO] Loading YOLOv8 model: {model_spec}")
    try:
        model = YOLO(model_spec)
        if apply_overrides and LABEL_OVERRIDES:
            model.names.update(LABEL_OVERRIDES)
    except Exception as exc:  # pragma: no cover - runtime dependency download
        print(f"[ERROR] Failed to load model: {exc}")
        sys.exit(1)

    print("[INFO] Model loaded successfully.")
    return model


def summarize_detections(result) -> str:
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "cls", None) is None:
        return "none"

    class_ids = boxes.cls.tolist() if hasattr(boxes, "cls") else []
    confidences = boxes.conf.tolist() if hasattr(boxes, "conf") and boxes.conf is not None else []

    if not class_ids:
        return "none"

    names = getattr(result, "names", {}) or {}
    entries = []
    for idx, cls_id in enumerate(class_ids):
        label = names.get(int(cls_id), str(int(cls_id))) if isinstance(names, dict) else str(int(cls_id))
        conf = None
        if confidences:
            conf = confidences[idx]
        if conf is not None:
            entries.append(f"{label} ({conf:.2f})")
        else:
            entries.append(label)
    return ", ".join(entries)


def run_inference(model: YOLO, source, conf: float):
    results = model.predict(source=source, conf=conf, verbose=False)
    if LABEL_OVERRIDES and results:
        for result in results:
            if getattr(result, "names", None):
                result.names.update(LABEL_OVERRIDES)
    if not results:
        return None
    return results[0]


def load_face_recognition_model():
    """Load InsightFace ArcFace model for face recognition."""
    if not INSIGHTFACE_AVAILABLE:
        print("[ERROR] InsightFace not installed. Run: pip install insightface onnxruntime")
        sys.exit(1)
    
    print("[INFO] Loading ArcFace face recognition model...")
    face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("[INFO] ArcFace model loaded successfully.")
    return face_app


def load_employee_database(face_app):
    """Load employee face embeddings from employee_db/ directory."""
    if not EMPLOYEE_DB_DIR.exists():
        print(f"[WARN] Employee database directory not found: {EMPLOYEE_DB_DIR}")
        print("[INFO] Create employee_db/<name>/ folders with 2-3 face images per person.")
        return {}
    
    employee_embeddings = {}
    employee_dirs = [d for d in EMPLOYEE_DB_DIR.iterdir() if d.is_dir()]
    
    if not employee_dirs:
        print(f"[WARN] No employee folders found in {EMPLOYEE_DB_DIR}")
        return {}
    
    print(f"[INFO] Loading employee database from {EMPLOYEE_DB_DIR}...")
    for person_dir in employee_dirs:
        person_name = person_dir.name
        embeddings = []
        
        image_files = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")) + list(person_dir.glob("*.jpeg"))
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            faces = face_app.get(img)
            if faces:
                embeddings.append(faces[0].embedding)
        
        if embeddings:
            # Average embeddings for this person
            avg_embedding = np.mean(embeddings, axis=0)
            employee_embeddings[person_name] = avg_embedding
            print(f"[INFO]   Loaded {len(embeddings)} face(s) for '{person_name}'")
    
    print(f"[INFO] Employee database loaded: {len(employee_embeddings)} people")
    return employee_embeddings


def recognize_face(face_app, employee_embeddings, person_crop, threshold=0.4):
    """
    Recognize a person's face from a cropped image.
    
    Returns:
        tuple: (name, similarity_score) or (None, 0.0) if no match
    """
    if not employee_embeddings:
        return None, 0.0
    
    faces = face_app.get(person_crop)
    if not faces:
        return None, 0.0
    
    query_embedding = faces[0].embedding
    
    best_match = None
    best_similarity = 0.0
    
    for name, db_embedding in employee_embeddings.items():
        # Cosine similarity
        similarity = np.dot(query_embedding, db_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding)
        )
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = name
    
    if best_similarity >= threshold:
        return best_match, best_similarity
    else:
        return "Unknown", best_similarity


def process_image(
    ppe_model: YOLO,
    person_model: Optional[YOLO],
    image_path: str,
    conf: float,
    show: bool,
    debug: bool,
    face_app=None,
    employee_embeddings=None,
) -> None:
    print(f"[INFO] Processing image: {image_path}")
    ppe_result = run_inference(ppe_model, image_path, conf)
    if ppe_result is None:
        print("[WARN] No PPE results returned by the model.")
    elif debug:
        print(f"[DEBUG] PPE detections: {summarize_detections(ppe_result)}")

    person_result = None
    if person_model is not None:
        person_result = run_inference(person_model, image_path, conf)
        if person_result is None:
            print("[WARN] No person detections returned by the model.")
        elif debug:
            print(f"[DEBUG] Person detections: {summarize_detections(person_result)}")

    if ppe_result is None and person_result is None:
        print("[WARN] No detections to display or save.")
        return

    annotated = (
        ppe_result.plot(img=cv2.imread(image_path).copy(), conf=False)
        if ppe_result is not None
        else cv2.imread(image_path)
    )
    if person_result is not None:
        annotated = person_result.plot(img=annotated.copy(), conf=False)
    
    # Face recognition on person boxes
    if face_app is not None and employee_embeddings is not None and person_result is not None:
        frame = cv2.imread(image_path)
        boxes = person_result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                # Add padding to capture face better
                h, w = frame.shape[:2]
                pad = 20
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size > 0:
                    name, similarity = recognize_face(face_app, employee_embeddings, person_crop)
                    if name:
                        label = f"{name} ({similarity:.2f})" if name != "Unknown" else f"Unknown ({similarity:.2f})"
                        if debug:
                            print(f"[DEBUG] Face recognition: {label}")
                        # Draw name above person box
                        cv2.putText(
                            annotated,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0) if name != "Unknown" else (0, 165, 255),
                            2,
                            cv2.LINE_AA,
                        )
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / timestamped_name("ppe_image", "jpg")
    cv2.imwrite(str(output_path), annotated)
    print(f"[INFO] Annotated image saved to: {output_path}")

    if show:
        window_title = "PPE Detection - Image"
        cv2.imshow(window_title, annotated)
        print("[INFO] Press any key in the image window to continue...")
        key = cv2.waitKey(0)
        if key == -1:
            # Ensure the message loop processes close events before destroying the window
            cv2.waitKey(1)
        try:
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(window_title)
        except cv2.error:
            cv2.destroyAllWindows()


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def process_video(
    ppe_model: YOLO,
    person_model: Optional[YOLO],
    source: Union[str, int],
    conf: float,
    show: bool,
    mode_label: str,
    debug: bool,
    face_app=None,
    employee_embeddings=None,
) -> None:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"[ERROR] Unable to open video source: {source}")
        sys.exit(1)

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = capture.get(cv2.CAP_PROP_FPS)
    fps_input = fps_input if fps_input and fps_input > 0 else 30.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / timestamped_name("ppe_video", "mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps_input, (frame_width, frame_height))

    print(f"[INFO] Processing {mode_label} stream. Press 'q' to quit.")
    print(f"[INFO] Saving annotated video to: {output_path}")

    prev_time = time.time()

    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            print("[INFO] End of stream or cannot fetch frame.")
            break

        ppe_result = run_inference(ppe_model, frame, conf)
        person_result = run_inference(person_model, frame, conf) if person_model is not None else None

        if ppe_result is None and person_result is None:
            annotated = frame
            if debug:
                print(f"[DEBUG] Frame {frame_index}: no detections")
        else:
            annotated = frame
            if ppe_result is not None:
                if debug:
                    print(f"[DEBUG] Frame {frame_index} PPE: {summarize_detections(ppe_result)}")
                annotated = ppe_result.plot(img=annotated.copy(), conf=False)
            if person_result is not None:
                if debug:
                    print(f"[DEBUG] Frame {frame_index} Person: {summarize_detections(person_result)}")
                annotated = person_result.plot(img=annotated.copy(), conf=False)
            
            # Face recognition on person boxes
            if face_app is not None and employee_embeddings is not None and person_result is not None:
                boxes = person_result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes.xyxy:
                        x1, y1, x2, y2 = map(int, box[:4])
                        # Add padding to capture face better
                        h, w = frame.shape[:2]
                        pad = 20
                        x1_crop = max(0, x1 - pad)
                        y1_crop = max(0, y1 - pad)
                        x2_crop = min(w, x2 + pad)
                        y2_crop = min(h, y2 + pad)
                        
                        person_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                        if person_crop.size > 0:
                            name, similarity = recognize_face(face_app, employee_embeddings, person_crop)
                            if name:
                                label = f"{name} ({similarity:.2f})" if name != "Unknown" else f"Unknown ({similarity:.2f})"
                                if debug and frame_index % 30 == 0:  # Print every 30 frames to avoid spam
                                    print(f"[DEBUG] Frame {frame_index} Face: {label}")
                                # Draw name above person box
                                cv2.putText(
                                    annotated,
                                    label,
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 255, 0) if name != "Unknown" else (0, 165, 255),
                                    2,
                                    cv2.LINE_AA,
                                )

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0.0
        prev_time = current_time
        draw_fps(annotated, fps)

        writer.write(annotated)

        if show:
            cv2.imshow(f"PPE Detection - {mode_label}", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] 'q' pressed. Exiting stream early.")
                break

        frame_index += 1

    capture.release()
    writer.release()
    if show:
        cv2.destroyWindow(f"PPE Detection - {mode_label}")

    print(f"[INFO] Finished processing. Output saved to: {output_path}")


def download_youtube_video(url: str) -> str:
    print(f"[INFO] Downloading video from YouTube: {url}")
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_template = str(VIDEOS_DIR / f"youtube_{timestamp}") + ".%(ext)s"

    ydl_opts = {
        "outtmpl": output_template,
        "format": "best[ext=mp4]/bestvideo[ext=mp4]/best",
        "noplaylist": True,
        "merge_output_format": "mp4",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_path = ydl.prepare_filename(info)
    except Exception as exc:
        print(f"[ERROR] Failed to download YouTube video: {exc}")
        sys.exit(1)

    print(f"[INFO] Video saved to: {downloaded_path}")
    return downloaded_path


def main() -> None:
    args = parse_args()
    youtube_url = getattr(args, "youtube_url", None)

    if youtube_url and args.mode != "video":
        print("[ERROR] --youtube-url can only be used with --mode video.")
        sys.exit(1)

    model_spec = resolve_model_spec(args.model)
    person_model_spec = resolve_model_spec(args.person_model) if args.person_model and args.person_model.lower() not in ("none", "") else None

    if args.mode == "video" and youtube_url:
        if yt_dlp is None:
            print("[ERROR] yt-dlp is not installed. Please install dependencies from requirements.txt.")
            sys.exit(1)
        source = download_youtube_video(youtube_url)
    else:
        source = validate_source(args.mode, args.source)

    ppe_model = load_model(model_spec, apply_overrides=True)
    person_model = load_model(person_model_spec) if person_model_spec else None

    # Initialize face recognition if enabled
    face_app = None
    employee_embeddings = None
    if args.enable_face_recognition:
        face_app = load_face_recognition_model()
        employee_embeddings = load_employee_database(face_app)
        if not employee_embeddings:
            print("[WARN] Face recognition enabled but no employee database found. Continuing without face recognition.")
            face_app = None

    if args.mode == "image":
        process_image(ppe_model, person_model, source, args.conf, args.show, args.debug, face_app, employee_embeddings)
    elif args.mode == "video":
        process_video(
            ppe_model,
            person_model,
            source,
            args.conf,
            args.show,
            mode_label="Video",
            debug=args.debug,
            face_app=face_app,
            employee_embeddings=employee_embeddings,
        )
    else:
      
        try:
            webcam_index = int(source)
        except ValueError:
            print("[ERROR] Webcam source must be an integer index (e.g., 0).")
            sys.exit(1)

        process_video(
            ppe_model,
            person_model,
            webcam_index,
            args.conf,
            args.show,
            mode_label="Webcam",
            debug=args.debug,
            face_app=face_app,
            employee_embeddings=employee_embeddings,
        )


if __name__ == "__main__":
    main()

