"""Export DINOv2 to ONNX for the production embedder. Run ONCE on a machine that has the weights
(e.g. a build box with access to your internal model mirror), then copy the .onnx into the
air-gapped environment. The service itself never downloads anything.

  pip install torch transformers onnx
  # put the model files (config.json, model.safetensors, preprocessor_config.json) in ./dinov2-small
  python tools/export_dinov2_onnx.py --model-dir ./dinov2-small --out models/dinov2/dinov2_vits14.onnx

Model sizes (DINOv2, Apache-2.0):  small ViT-S/14 (384-d, ~22M params)  base ViT-B/14 (768-d, ~86M)
ViT-S is plenty for card identification and runs in a few ms per image on a T4.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="local folder with the Hugging Face DINOv2 files")
    ap.add_argument("--out", default="models/dinov2/dinov2_vits14.onnx")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--fp16", action="store_true", help="convert weights to float16 (T4 tensor cores)")
    a = ap.parse_args()

    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(a.model_dir, local_files_only=True).eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            out = self.m(pixel_values=pixel_values)
            cls = out.last_hidden_state[:, 0]                      # CLS token
            patch_mean = out.last_hidden_state[:, 1:].mean(dim=1)   # average of patch tokens
            return torch.cat([cls, patch_mean], dim=-1)             # robust global descriptor

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, a.size, a.size)
    torch.onnx.export(Wrapper(model), dummy, str(out), input_names=["pixel_values"], output_names=["embedding"],
                      dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
                      # opset 18: the torch>=2.9 exporter emits 18 natively and its down-conversion to 17
                      # fails on ReduceMean; onnxruntime>=1.17 supports 18. Keep weights inside the .onnx
                      # (model is <2GB) so the air-gapped copy is a single file.
                      opset_version=18, external_data=False)
    if a.fp16:
        import onnx
        from onnxconverter_common import float16
        m = float16.convert_float_to_float16(onnx.load(str(out)), keep_io_types=True)
        onnx.save(m, str(out))
    print(f"exported {out}  — set embedder.kind=dinov2_onnx and embedder.onnx_path={out}, "
          "then re-embed the references (python tools/sync_registry.py --reembed, or POST /v1/registry/sync?reembed=true) "
          "and re-run tools/evaluate.py "
          "to recalibrate embedder.similarity_floor.")


if __name__ == "__main__":
    main()
