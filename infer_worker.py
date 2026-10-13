"""Runs PyTorch in its own process, on that process's MAIN thread.
(On Python 3.15, PyTorch crashes when it creates tensors on background threads.)

Protocol over a multiprocessing Pipe:
  child -> parent : {"ok": True, "class_names": [...], "device": "cpu"|"cuda"}   (once, at startup)
  parent -> child : 2304 raw bytes (48x48 grayscale face)
  child -> parent : list of class probabilities, or {"error": "..."}
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def run(conn, checkpoint_path):
    try:
        torch.set_num_threads(1)
        from src.models.resnet import EmotionResNet

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        class_names = checkpoint.get("class_names")
        if not class_names:
            raise ValueError("The checkpoint is missing class_names metadata.")
        model_name = checkpoint.get("model_name", "resnet")
        if model_name != "resnet":
            raise ValueError(f"Expected a ResNet checkpoint, got {model_name}.")
        model = EmotionResNet(num_classes=len(class_names))
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()

        def infer(data: bytes):
            t = torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(1, 1, 48, 48).float()
            t = ((t / 255.0 - 0.5) / 0.5).to(device)
            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    logits = model(t)
                probs = F.softmax(logits.float(), dim=1)[0]
            return probs.tolist()

        infer(bytes(2304))  # warm-up so the first real frame isn't slow
        conn.send({"ok": True, "class_names": list(class_names), "device": device.type})
    except Exception as exc:
        conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return

    while True:
        try:
            data = conn.recv_bytes()
        except (EOFError, OSError):
            break
        if not data:
            break
        try:
            conn.send(infer(data))
        except Exception as exc:
            conn.send({"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    # Launched by app.py as:  python infer_worker.py <port> <authkey_hex> <checkpoint_path>
    from multiprocessing.connection import Client

    sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `src` is importable
    port = int(sys.argv[1])
    authkey = bytes.fromhex(sys.argv[2])
    checkpoint = sys.argv[3]
    connection = Client(("127.0.0.1", port), authkey=authkey)
    run(connection, checkpoint)