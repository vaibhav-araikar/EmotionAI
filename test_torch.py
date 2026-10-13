import subprocess, sys

tests = {
    "1. torch, main thread": (
        "import torch\n"
        "t = torch.frombuffer(bytearray(2304), dtype=torch.uint8).reshape(1,1,48,48).float()\n"
        "print(t.sum())"
    ),
    "2. torch, worker thread": (
        "import torch, threading\n"
        "def f():\n"
        "    t = torch.frombuffer(bytearray(2304), dtype=torch.uint8).reshape(1,1,48,48).float()\n"
        "    print(t.sum())\n"
        "th = threading.Thread(target=f); th.start(); th.join()"
    ),
    "3. cv2 first, then torch in worker thread": (
        "import cv2, torch, threading\n"
        "def f():\n"
        "    t = torch.frombuffer(bytearray(2304), dtype=torch.uint8).reshape(1,1,48,48).float()\n"
        "    print(t.sum())\n"
        "th = threading.Thread(target=f); th.start(); th.join()"
    ),
    "4. torch first, then cv2, worker thread": (
        "import torch, cv2, threading\n"
        "def f():\n"
        "    t = torch.frombuffer(bytearray(2304), dtype=torch.uint8).reshape(1,1,48,48).float()\n"
        "    print(t.sum())\n"
        "th = threading.Thread(target=f); th.start(); th.join()"
    ),
}

for name, code in tests.items():
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    status = "OK" if r.returncode == 0 else f"CRASH (exit code {r.returncode})"
    print(f"{name:45s} -> {status}")
    if r.returncode != 0 and r.stderr.strip():
        print("   ", r.stderr.strip().splitlines()[-1][:200])