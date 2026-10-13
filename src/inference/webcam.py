"""Live webcam emotion demo. Uses OpenCV Haar face detection and the trained model."""
import argparse, cv2, torch
from PIL import Image
from src.inference.predict import load_model
from src.data.dataset import EMOTION_LABELS
from src.data.transforms import build_transforms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "resnet"], default="cnn")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    transform = build_transforms(False)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions or --camera index.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            for x, y, w, h in faces:
                face = gray[y:y+h, x:x+w]
                if face.size == 0: continue
                face_img = Image.fromarray(cv2.resize(face, (48, 48)))
                tensor = transform(face_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=1)[0]
                idx = int(probs.argmax().item())
                label = f"{EMOTION_LABELS[idx]} {probs[idx].item()*100:.0f}%"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, max(25, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Facial Emotion Recognition - press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    finally:
        cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
