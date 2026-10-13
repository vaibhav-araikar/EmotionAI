import subprocess, sys

ops = {
    "torch.zeros": "torch.zeros(1,1,48,48)",
    "torch.tensor(list)": "torch.tensor([0.0]*2304).reshape(1,1,48,48)",
    "zeros + arithmetic": "(torch.zeros(1,1,48,48) + 1).sum()",
    "conv2d (model-like)": "torch.nn.functional.conv2d(torch.zeros(1,1,48,48), torch.zeros(4,1,3,3))",
    "frombuffer": "torch.frombuffer(bytearray(2304), dtype=torch.uint8).float()",
}

for name, expr in ops.items():
    code = (
        "import torch, threading\n"
        "def f():\n"
        f"    r = {expr}\n"
        "    print('ok')\n"
        "th = threading.Thread(target=f); th.start(); th.join()\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(f"{name:25s} -> {'OK' if r.returncode == 0 else 'CRASH'}")