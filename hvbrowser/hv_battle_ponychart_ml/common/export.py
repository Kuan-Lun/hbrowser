"""ONNX model export utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_onnx(model: nn.Module, output_path: Path) -> None:
    """Export a PyTorch model to ONNX format."""
    import warnings

    import onnx

    model.eval()
    model_cpu = model.cpu()
    dummy = torch.randn(1, 3, 224, 224)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Missing annotation for parameter")
        torch.onnx.export(
            model_cpu,
            (dummy,),
            str(output_path),
            input_names=["input"],
            output_names=["logits"],
            opset_version=18,
        )
    # Merge external data into single file if needed
    external_data = Path(str(output_path) + ".data")
    if external_data.exists():
        onnx_model = onnx.load(str(output_path), load_external_data=True)
        onnx.save_model(
            onnx_model,
            str(output_path),
            save_as_external_data=False,
        )
        external_data.unlink()
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("ONNX model exported: %s (%.1f MB)", output_path, size_mb)
