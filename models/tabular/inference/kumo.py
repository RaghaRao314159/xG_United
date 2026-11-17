"""Kumo Tabular members of the tabular models, on a CUDA GPU (default) or the CPU.

Kumo Tabular (`pip install structured-data-models`, NVIDIA's `nvidia/Kumo-Tabular` checkpoint) is a
pretrained in-context model: nothing in it was trained by us. A Kumo member of Ragha's tabular
models is fully described by its *context*, the labelled training rows handed to `fit`, and by the
subset of feature columns it reads; both are files under weights/tabular/.

On a GPU the members run exactly as they were built: float16 autocast, the same seed and the
same call order (fit once, then predict the rows, then their home/away-mirrored copies). On the
CPU they run in float32, which is slow (hours for the whole match) and gives probabilities that
differ from the GPU's in about the third decimal place.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from main_pipeline.config import torch_device

ESTIMATORS = 4                  # Kumo's internal ensemble size, as when the members were built
SEED = 0
BATCH_ROWS = 8192               # query rows per predict call, as when the members were built
CPU_CHUNK_BYTES = 2 * 1024**3   # memory one chunk of an attention layer may use on the CPU


def chunk_attention_on_cpu() -> None:
    """Make `sdm` split its attention layers into memory-sized chunks on the CPU too.

    On a GPU, `sdm` splits each attention layer's batch of independent rows into chunks that fit
    5% of the GPU's memory. On a CPU it does not split at all, which needs more than 30 GB for a
    50,000-row context. This applies the GPU rule with a fixed budget. The rows of a batch are
    computed independently, so the split does not change any answer.
    """
    from sdm.nn.attention import TransformerBlock

    def batch_size_limit(block, device, element_size, query_length, key_value_length=None) -> int:
        per_row = block.peak_bytes_per_example(element_size=element_size, query_length=query_length,
                                               key_value_length=key_value_length)
        return min(max(CPU_CHUNK_BYTES // max(per_row, 1), 1), 65_535)

    TransformerBlock.auto_batch_size_limit = batch_size_limit


class KumoMember:
    """One in-context Kumo classifier: its context set and the columns it reads."""

    def __init__(self, context_file: Path, columns_file: Path, n_classes: int) -> None:
        context = np.load(context_file)
        self.x_context = _finite_or_nan(context["x_context"])
        self.y_context = context["y_context"].reshape(-1).astype(np.int64)
        self.columns = np.load(columns_file)
        self.n_classes = n_classes

    def predict_twice(self, x: np.ndarray, x_mirrored: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Class probabilities, shape (n, n_classes) in class-code order, for the rows of `x`
        and then for their mirrored copies. Both use one fit, in the original call order."""
        import torch
        from sdm.models import KumoTabular

        device = torch_device()
        if device.type == "cpu":
            chunk_attention_on_cpu()
        model = KumoTabular(task="classification", size="large", device=device)
        torch.manual_seed(SEED)
        with _precision(device):
            model.fit(torch.from_numpy(self.x_context).to(device),
                      torch.from_numpy(self.y_context[:, None]).to(device), num_estimators=ESTIMATORS)
        try:
            return self._predict(model, x, device), self._predict(model, x_mirrored, device)
        finally:
            model.clear()

    def _predict(self, model, x: np.ndarray, device) -> np.ndarray:
        import torch
        from sdm import Stype

        rows = _finite_or_nan(x[:, self.columns])
        out = []
        with _precision(device):
            for start in range(0, len(rows), BATCH_ROWS):
                table = model.predict(torch.from_numpy(rows[start:start + BATCH_ROWS]).to(device))
                names = list(table.columns[Stype.numerical])    # class order varies between calls
                values = table.numerical.float().cpu().numpy()
                block = np.zeros((len(values), self.n_classes))
                for code in range(self.n_classes):
                    if str(code) in names:
                        block[:, code] = values[:, names.index(str(code))]
                out.append(block)
        return np.concatenate(out)


def _precision(device):
    """float16 autocast on a GPU, as the members were built; plain float32 on the CPU."""
    import torch
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _finite_or_nan(x: np.ndarray) -> np.ndarray:
    """float32 copy with infinities replaced by NaN, which Kumo treats as missing."""
    return np.where(np.isfinite(x), x, np.nan).astype(np.float32)
