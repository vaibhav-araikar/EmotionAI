import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import faulthandler
faulthandler.enable()

import atexit
import secrets
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Listener
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
from av import VideoFrame

# PyTorch lives in a separate process (infer_worker.py); this app never imports torch.
WORKER_PATH = ROOT_DIR / "infer_worker.py"
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "resnet_best.pth"
st.set_page_config(page_title="EmotionAI | Facial Expression Recognition", page_icon="😊", layout="wide")

# label, emoji, description
EMOTION_INFO = {
    "angry": ("Angry", "😠", "A visibly angry facial expression."),
    "disgust": ("Disgust", "🤢", "A visibly disgusted facial expression."),
    "fear": ("Fear", "😨", "A visibly fearful facial expression."),
    "happy": ("Happy", "😄", "A visibly happy facial expression."),
    "neutral": ("Neutral", "😐", "A relatively neutral facial expression."),
    "sad": ("Sad", "😢", "A visibly sad facial expression."),
    "surprise": ("Surprise", "😲", "A visibly surprised facial expression."),
}
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


def to_gray48_bytes(pil_image):
    """Grayscale + bilinear resize to 48x48, as raw bytes (same preprocessing as training)."""
    return pil_image.convert("L").resize((48, 48), Image.BILINEAR).tobytes()


class InferenceClient:
    """Talks to the PyTorch process. Thread-safe."""

    def __init__(self, checkpoint_path):
        authkey = secrets.token_bytes(16)
        listener = Listener(("127.0.0.1", 0), authkey=authkey)
        port = listener.address[1]
        env = {**os.environ, "PYTHONPATH": str(ROOT_DIR)}
        self.proc = subprocess.Popen(
            [sys.executable, str(WORKER_PATH), str(port), authkey.hex(), str(checkpoint_path)],
            cwd=str(ROOT_DIR),
            env=env,
        )
        atexit.register(self.close)
        self.lock = threading.Lock()

        # Wait for the model process to connect (without hanging forever if it dies).
        result = {}

        def _accept():
            try:
                result["conn"] = listener.accept()
            except Exception as exc:  # noqa: BLE001
                result["err"] = exc

        t = threading.Thread(target=_accept, daemon=True)
        t.start()
        deadline = time.time() + 180
        while t.is_alive() and time.time() < deadline:
            t.join(0.5)
            if self.proc.poll() is not None and "conn" not in result:
                break
        if "conn" not in result:
            code = self.proc.poll()
            self.close()
            listener.close()
            raise RuntimeError(
                f"Model process failed to start (exit code {code}). See the terminal for details."
            )
        listener.close()
        self.conn = result["conn"]

        if not self.conn.poll(180):
            self.close()
            raise RuntimeError("Timed out waiting for the model to load.")
        info = self.conn.recv()
        if not info.get("ok"):
            self.close()
            raise RuntimeError(info.get("error", "Model process failed to start."))
        self.class_names = info["class_names"]
        self.device = info["device"]

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass

    def is_alive(self):
        return self.proc.poll() is None

    def predict_probs(self, gray48_bytes):
        with self.lock:
            self.conn.send_bytes(gray48_bytes)
            if not self.conn.poll(15):
                raise TimeoutError("Model process did not answer in time.")
            result = self.conn.recv()
        if isinstance(result, dict):
            raise RuntimeError(result.get("error", "Unknown inference error"))
        return result


@st.cache_resource
def get_client():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    return InferenceClient(CHECKPOINT_PATH)


def predict_expression(image, client):
    image = ImageOps.exif_transpose(image).convert("RGB")
    probs = client.predict_probs(to_gray48_bytes(image))
    scores = {client.class_names[i]: float(probs[i]) for i in range(len(client.class_names))}
    return max(scores, key=scores.get), scores


class EmotionVideoProcessor(VideoProcessorBase):
    """recv() only draws the latest result. Face detection runs in a background thread and
    the model runs in the separate PyTorch process, so the video never stalls."""

    def __init__(self, client):
        self.client = client
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.lock = threading.Lock()
        self.new_frame = threading.Event()
        self.stop_flag = threading.Event()
        self.latest_gray = None

        self.frame_count = 0
        self.process_every = 3      # hand a frame to the worker every N frames
        self.det_width = 320        # small detection image = fast
        self.max_misses = 4         # keep the last box for a few missed detections (less flicker)
        self.misses = 0

        # Results (full-frame coordinates)
        self.last_box = None
        self.last_emotion = None
        self.last_confidence = 0.0
        self.last_error = None
        self.emoji_patches = {}

        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def on_ended(self):
        self.stop_flag.set()

    # ---------- background thread: detection, then ask the model process ----------
    def _worker(self):
        while not self.stop_flag.is_set():
            if not self.new_frame.wait(timeout=0.5):
                continue
            self.new_frame.clear()
            with self.lock:
                if self.latest_gray is None:
                    continue
                gray, scale = self.latest_gray
            try:
                faces = self.detector.detectMultiScale(
                    gray, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30)
                )
                if len(faces) == 0:
                    self.misses += 1
                    if self.misses >= self.max_misses:
                        with self.lock:
                            self.last_box = None
                            self.last_emotion = None
                            self.last_confidence = 0.0
                    continue

                self.misses = 0
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                face = Image.fromarray(np.ascontiguousarray(gray[y:y + h, x:x + w]))
                probs = self.client.predict_probs(to_gray48_bytes(face))
                idx = int(np.argmax(probs))
                with self.lock:
                    self.last_box = (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
                    self.last_emotion = self.client.class_names[idx]
                    self.last_confidence = float(probs[idx])
                    self.last_error = None
            except Exception as exc:
                with self.lock:
                    self.last_error = f"{type(exc).__name__}: {exc}"

    # ---------- emoji: render once, blend only the small region ----------
    def _emoji_patch(self, emoji):
        if emoji not in self.emoji_patches:
            patch = Image.new("RGBA", (42, 42), (0, 0, 0, 0))
            try:
                font = ImageFont.truetype(r"C:\Windows\Fonts\seguiemj.ttf", 30)
                ImageDraw.Draw(patch).text((2, 1), emoji, font=font, embedded_color=True)
            except Exception:
                pass  # no emoji font available -> blank patch; the text label still shows
            self.emoji_patches[emoji] = np.array(patch)
        return self.emoji_patches[emoji]

    def _draw_emoji(self, frame_bgr, emoji, x, y):
        patch = self._emoji_patch(emoji)
        h, w = patch.shape[:2]
        x = max(0, min(x, frame_bgr.shape[1] - w))
        y = max(0, min(y, frame_bgr.shape[0] - h))
        roi = frame_bgr[y:y + h, x:x + w]
        alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
        patch_bgr = patch[:, :, 2::-1]  # RGB -> BGR
        roi[:] = (patch_bgr * alpha + roi * (1.0 - alpha)).astype(np.uint8)

    # ---------- fast path: runs for every video frame ----------
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        H, W = img.shape[:2]

        if self.frame_count % self.process_every == 0 and not self.new_frame.is_set():
            scale = min(1.0, self.det_width / float(W))
            if scale < 1.0:
                small = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
            else:
                small = img
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            with self.lock:
                self.latest_gray = (gray, scale)
            self.new_frame.set()

        with self.lock:
            box = self.last_box
            emotion = self.last_emotion
            conf = self.last_confidence
            err = self.last_error

        if box and emotion:
            x, y, w, h = box
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))
            cv2.rectangle(img, (x, y), (x + w, y + h), (50, 220, 100), 2)
            label, emoji, _ = EMOTION_INFO.get(emotion, (emotion.title(), "🙂", ""))
            text = f"{label} {conf:.0%}"
            ty = max(28, y - 8)
            cv2.putText(img, text, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (50, 255, 100), 2, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            self._draw_emoji(img, emoji, x + tw + 8, max(0, ty - 34))
        elif err:
            cv2.putText(img, "Prediction error - check terminal", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "Move your face into the frame", (18, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)

        return VideoFrame.from_ndarray(img, format="bgr24")


st.markdown("""
<style>
.main-title {font-size:2.6rem;font-weight:800;letter-spacing:-1px;margin-bottom:0}
.subtitle {font-size:1.05rem;opacity:.75;margin-top:.3rem;margin-bottom:1.5rem}
.result-card {padding:1.3rem;border-radius:14px;border:1px solid rgba(128,128,128,.3);margin-top:.8rem}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="main-title">😊 EmotionAI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Facial Expression Recognition powered by Deep Learning</div>', unsafe_allow_html=True)

with st.sidebar:
    st.title("About EmotionAI")
    st.write("EmotionAI uses a trained ResNet-style deep learning model to classify visible facial expressions into seven categories.")
    st.divider()
    st.subheader("Model information")
    st.write("Architecture: ResNet-style CNN")
    st.write("Input: 48 × 48 grayscale face")
    st.write("Classes: 7")
    st.caption("Predictions describe visible facial expressions; they cannot reliably determine a person's internal emotional state.")

try:
    client = get_client()
    if not client.is_alive():
        get_client.clear()
        client = get_client()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.info("Check that checkpoints/resnet_best.pth exists and launch Streamlit from your project folder.")
    st.stop()

class_names = client.class_names

m1, m2, m3 = st.columns(3)
m1.metric("Model", "ResNet")
m2.metric("Emotion classes", len(class_names))
m3.metric("Inference device", "GPU" if client.device == "cuda" else "CPU")
st.divider()
live_tab, upload_tab, photo_tab = st.tabs(["🎥 Live emotion detection", "📁 Upload an image", "📷 Capture a photo"])

with live_tab:
    st.subheader("Live facial expression detection")
    st.write("Click START, allow camera access, and look toward the camera. The video overlay displays the detected expression, emoji, and confidence score.")
    st.info("Use good front lighting and keep your face visible. If the camera does not start, check browser camera permissions. Camera access generally requires localhost or HTTPS.")
    webrtc_ctx = webrtc_streamer(
        key="emotion-live-camera",
        video_processor_factory=lambda: EmotionVideoProcessor(client),
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {"frameRate": {"ideal": 15}, "width": {"ideal": 640}, "height": {"ideal": 480}},
            "audio": False,
        },
        async_processing=True,
    )
    if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
        processor_error = webrtc_ctx.video_processor.last_error
        if processor_error:
            st.warning(processor_error)

with upload_tab:
    st.subheader("Analyze a facial expression")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], help="Use a clear, front-facing face image for best results.", key="upload-image")
    if uploaded_file is not None:
        try:
            image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
            left, right = st.columns(2, gap="large")
            with left:
                st.image(image, caption="Input image", width="stretch")
            with right:
                with st.spinner("Analyzing facial expression..."):
                    predicted, scores = predict_expression(image, client)
                label, emoji, description = EMOTION_INFO.get(predicted, (predicted.title(), "🙂", ""))
                confidence = scores[predicted]
                st.markdown(f'<div class="result-card"><h1>{emoji} {label}</h1><p>{description}</p><h3>{confidence:.1%} model confidence</h3></div>', unsafe_allow_html=True)
                st.progress(confidence)
                st.caption("Confidence is the model's output score, not a calibrated probability or guarantee of correctness.")
            st.divider()
            st.subheader("Scores for all expressions")
            for emotion, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
                label, emoji, _ = EMOTION_INFO.get(emotion, (emotion.title(), "🙂", ""))
                st.write(f"**{emoji} {label}** — {score:.1%}")
                st.progress(score)
        except Exception as exc:
            st.error(f"Could not analyze this image: {exc}")

with photo_tab:
    st.subheader("Take a picture")
    st.write("Turn on the camera below, position your face in the frame, and capture a photo to analyze it.")
    st.caption("The camera is off by default so it doesn't conflict with the live detection stream. Stop the live stream before enabling it.")
    enable_photo = st.checkbox("Enable photo camera", value=False, key="enable-photo-camera")
    if enable_photo:
        camera_image = st.camera_input("Capture a face image", key="capture-photo")
        if camera_image is not None:
            try:
                image = ImageOps.exif_transpose(Image.open(camera_image)).convert("RGB")
                predicted, scores = predict_expression(image, client)
                label, emoji, description = EMOTION_INFO.get(predicted, (predicted.title(), "🙂", ""))
                left, right = st.columns(2)
                with left:
                    st.image(image, caption="Captured image", width="stretch")
                with right:
                    st.markdown(f"## {emoji} {label}")
                    st.write(description)
                    st.metric("Model confidence", f"{scores[predicted]:.1%}")
                    st.subheader("Expression scores")
                    for emotion, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
                        score_label, score_emoji, _ = EMOTION_INFO.get(emotion, (emotion.title(), "🙂", ""))
                        st.write(f"**{score_emoji} {score_label}** — {score:.1%}")
                        st.progress(score)
            except Exception as exc:
                st.error(f"Could not analyze the captured image: {exc}")

st.divider()
st.caption("EmotionAI · Deep Learning Project · PyTorch + Streamlit")