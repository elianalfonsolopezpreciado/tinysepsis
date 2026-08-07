"""Export the trained TinySepsis GRU to ONNX and benchmark CPU/GPU latency."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tinysepsis.models.tiny_sepsis import TinySepsisModel  # noqa: E402

CKPT_PATH = ROOT / "results" / "checkpoints" / "tinysepsis_best.pt"
ONNX_PATH = ROOT / "results" / "checkpoints" / "tinysepsis.onnx"
LATENCY_PATH = ROOT / "results" / "checkpoints" / "latency_benchmark.json"


def main():
    device = torch.device("cpu")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    args = ckpt["args"]

    model = TinySepsisModel(
        num_dynamic_features=ckpt["num_dynamic"],
        num_static_features=ckpt["num_static"],
        hidden_size=args["hidden_size"],
        num_layers=args["num_layers"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    seq_len = args["seq_len"]
    dummy_seq = torch.randn(1, seq_len, ckpt["num_dynamic"])
    dummy_pad = torch.ones(1, seq_len)
    dummy_static = torch.randn(1, ckpt["num_static"])

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_seq, dummy_pad, dummy_static),
        str(ONNX_PATH),
        input_names=["seq", "pad_mask", "static"],
        output_names=["logit"],
        dynamic_axes={"seq": {0: "batch"}, "pad_mask": {0: "batch"}, "static": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported ONNX model to {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.2f} MB)", flush=True)

    # --- Numerical parity check: ONNX Runtime output must match PyTorch ---
    verify_sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    check_seq = torch.randn(3, seq_len, ckpt["num_dynamic"])
    check_pad = torch.ones(3, seq_len)
    check_static = torch.randn(3, ckpt["num_static"])
    with torch.no_grad():
        torch_out = model(check_seq, check_pad, check_static).numpy()
    onnx_out = verify_sess.run(None, {
        "seq": check_seq.numpy().astype(np.float32),
        "pad_mask": check_pad.numpy().astype(np.float32),
        "static": check_static.numpy().astype(np.float32),
    })[0]
    max_abs_diff = float(np.max(np.abs(torch_out - onnx_out.squeeze())))
    print(f"Parity check: max |PyTorch - ONNX| = {max_abs_diff:.6f}", flush=True)
    if max_abs_diff > 1e-3:
        raise RuntimeError(f"ONNX export parity check FAILED (diff={max_abs_diff}) -- exported model does not match PyTorch")
    n_onnx_params = sum(int(np.prod(t.dims)) for t in onnx.load(str(ONNX_PATH)).graph.initializer)
    print(f"ONNX initializer parameter count: {n_onnx_params:,} (PyTorch: {model.num_parameters():,})", flush=True)
    if n_onnx_params < model.num_parameters() * 0.9:
        raise RuntimeError("ONNX graph is missing weights (likely GRU flat_weights not captured) -- export is broken")

    # --- Latency benchmark: PyTorch CPU vs ONNX Runtime CPU, single-sample inference ---
    n_warmup, n_runs = 10, 200

    def bench_torch():
        with torch.no_grad():
            for _ in range(n_warmup):
                model(dummy_seq, dummy_pad, dummy_static)
            t0 = time.perf_counter()
            for _ in range(n_runs):
                model(dummy_seq, dummy_pad, dummy_static)
            return (time.perf_counter() - t0) / n_runs * 1000

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    inputs = {
        "seq": dummy_seq.numpy().astype(np.float32),
        "pad_mask": dummy_pad.numpy().astype(np.float32),
        "static": dummy_static.numpy().astype(np.float32),
    }

    def bench_onnx():
        for _ in range(n_warmup):
            sess.run(None, inputs)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            sess.run(None, inputs)
        return (time.perf_counter() - t0) / n_runs * 1000

    torch_ms = bench_torch()
    onnx_ms = bench_onnx()

    n_params = model.num_parameters()
    result = {
        "n_parameters": n_params,
        "onnx_file_size_mb": ONNX_PATH.stat().st_size / 1e6,
        "pytorch_cpu_latency_ms": torch_ms,
        "onnxruntime_cpu_latency_ms": onnx_ms,
        "seq_len": seq_len,
        "hidden_size": args["hidden_size"],
    }
    with open(LATENCY_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
