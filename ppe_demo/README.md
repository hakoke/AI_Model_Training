# PPE Detection Demo

Lightweight YOLOv8 PPE detection project with optional ArcFace face recognition. Drop images or videos into `input/` and run detection in seconds.

## Quick Start Commands

Activate environment:
```powershell
..\venv\Scripts\Activate.ps1
```

### Basic PPE Detection
```bash
# Image detection
python main.py --mode image --source input/images/test.png --conf 0.3 --debug

# Video detection
python main.py --mode video --source input/videos/test.mp4 --conf 0.3 --debug

# YouTube video
python main.py --mode video --youtube-url https://www.youtube.com/shorts/_nm1Yb8sxxY --conf 0.3 --debug

# Webcam (live)
python main.py --mode webcam --conf 0.3
```

### With Face Recognition
```bash
# Enable ArcFace face recognition (requires employee_db/ setup - see below)
python main.py --mode image --source input/images/test.png --conf 0.3 --enable-face-recognition --debug

# Face recognition on video
python main.py --mode video --source input/videos/test.mp4 --conf 0.3 --enable-face-recognition
```

Add `--show False` to skip the display window and only save outputs.

## Features
- **PPE Detection:** Helmets, vests, masks, gloves, face shields, ear protection, safety equipment
- **Dual-Model Architecture:** 
  - Primary: `best_all6.pt` (YOLOv8m) for PPE detection
  - Secondary: `yolov8n.pt` for robust person detection
- **Face Recognition (Optional):** ArcFace embeddings for employee identification/attendance
- Works with images, videos, or live webcam feed
- Saves annotated outputs automatically to `output/`
- Optional GUI display, configurable confidence threshold, and FPS overlay for video/webcam
- Download YouTube videos straight into `input/videos` with one flag

## Project structure
```
ppe_demo/
├── main.py
├── input/
│   ├── images/
│   └── videos/
├── output/
├── models/
│   └── best_all6.pt         # Your fine-tuned PPE model
├── employee_db/              # (Optional) Face recognition database
│   ├── person1/
│   │   ├── face1.jpg
│   │   └── face2.jpg
│   └── person2/
│       ├── face1.jpg
│       └── face2.jpg
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If you hit OpenCV codec issues, install:
```bash
pip install opencv-python-headless
```

### Face Recognition Setup (Optional)
To enable face recognition for employee identification:

1. **Install dependencies** (if not already done):
   ```bash
   pip install insightface onnxruntime
   ```

2. **Create employee database:**
   ```
   employee_db/
   ├── john_smith/
   │   ├── photo1.jpg
   │   └── photo2.jpg
   ├── jane_doe/
   │   ├── photo1.jpg
   │   └── photo2.jpg
   └── ...
   ```
   - Create a folder for each employee named with their identifier (e.g., `john_smith`, `jane_doe`)
   - Add 2-3 clear face photos per person (different angles recommended)
   - Supported formats: `.jpg`, `.png`, `.jpeg`

3. **Run with face recognition enabled:**
   ```bash
   python main.py --mode image --source input/images/test.png --enable-face-recognition
   ```

4. **How it works:**
   - The model detects people using person boxes
   - ArcFace extracts face embeddings from each detected person
   - Embeddings are compared against your employee database
   - Names appear above person boxes with confidence scores
   - Unknown faces show as "Unknown (0.XX)"

5. **Testing without employees:**
   - To test if face recognition works, take 2 selfies and put them in `employee_db/test_person/`
   - Run detection on another photo with yourself in it
   - If it labels you correctly, the system is working!

## Usage
```bash
python main.py --mode image --source input/images/test.jpg
python main.py --mode video --source input/videos/warehouse.mp4
python main.py --mode webcam
```

### Optional flags
- `--conf 0.5` sets the detection confidence threshold (default 0.5, lower for more detections)
- `--show False` disables the GUI window (while still saving outputs)
- `--source` can be a webcam index (e.g., `--source 1`) when `--mode webcam`
- `--youtube-url <link>` downloads a YouTube video (video mode only) before running detection
- `--debug` prints detection summaries frame-by-frame (handy when nothing shows up)
- `--model custom.pt` to load your own checkpoint (default is `models/best_all6.pt`)
- `--person-model yolov8m.pt` swaps the model used for people (default `yolov8n.pt` auto-downloads), use `None` to disable
- `--enable-face-recognition` activates ArcFace face recognition (requires `employee_db/` setup)

### Output naming
- Image results → `ppe_image_<timestamp>.jpg`
- Video/Webcam results → `ppe_video_<timestamp>.mp4`

## Tips
- Press `q` to quit video or webcam streams early.
- Outputs land in the `output/` directory. Delete old runs when you want to start fresh.
- Lower the confidence threshold with `--conf 0.3` or `--conf 0.1` if detections are missing PPE items.
- Use `--youtube-url` in video mode to grab sample footage quickly. Videos land in `input/videos` automatically.
- Person detections rely on the Ultralytics general model (`yolov8n.pt` by default). Replace it via `--person-model` if you need a different backbone.
- First run expects PPE weights at `models/best_all6.pt`. Either drop in your Ultralytics PPE export or supply `--model` pointing to another `.pt` file.
- Swap in a different model by pointing `--model` at another Ultralytics ID or local `.pt` file whenever you fine-tune (e.g., HUB export).
- Add `--show False` if you just want the saved output (`output/ppe_video_<timestamp>.mp4`) without the live window.
- YouTube downloads pick a single MP4 stream, so FFmpeg is optional.
- For face recognition, 2-3 clear face photos per employee is sufficient for ~85-95% accuracy.

## Model Details
- **PPE Model:** `best_all6.pt` (YOLOv8m fine-tuned on 5 PPE datasets)
  - Classes: person, helmet_on/off, vest_on/off, mask_on/off, gloves_on/off, safety_cone, machinery, vehicle, face_shield, ear_protection, hands
- **Person Model:** `yolov8n.pt` (COCO pretrained, 80 classes including robust person detection)
- **Face Recognition:** InsightFace ArcFace (512-D embeddings, cosine similarity matching)

## Troubleshooting
- **No detections:** Lower `--conf` to 0.3 or 0.1
- **Missing masks/gloves:** These classes have sparse training data; fine-tune with more examples or use lower confidence
- **Face recognition not working:** Ensure `employee_db/` structure is correct and images contain clear faces
- **Slow performance:** Face recognition adds overhead; for real-time needs, skip it or run on GPU
