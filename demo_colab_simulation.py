import torch
from tice_multi_agent_sim_quantum_sentinel_plus import run_sim, save_dashboard_png

def demo_run(device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    res = run_sim(epochs=5, agents=5, adversarial_flip_rate=0.2, device=device)
    print(f"[{device.upper()}] Proof coherence: {res.proof_coherence}")
    print(f"[{device.upper()}] Λ values: {res.lambdas}")
    if res.phi_ciphertext is not None:
        print(f"[{device.upper()}] Encrypted Φ bytes: {len(res.phi_ciphertext)}")
    path = save_dashboard_png(res, f"dashboard_{device}.png")
    if path:
        print(f"[{device.upper()}] Dashboard saved: {path}")
    return res

# CPU demo
cpu_res = demo_run("cpu")

# GPU demo (if available)
if torch.cuda.is_available():
    gpu_res = demo_run("cuda")
else:
    print("CUDA not available; GPU demo skipped.")
