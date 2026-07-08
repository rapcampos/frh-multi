"""Regression test pinning generate_with_topk_guide to the golden reference.

The golden file is produced by 14_golden_reference.ipynb and is only valid on
the machine/GPU that generated it (AWQ inference is hardware-local). Run with:

    uv run pytest -m gpu tests/test_golden.py
"""

import json
from pathlib import Path

import pytest

GOLDEN_PATH = Path(__file__).parent / "golden" / "single_concept.json"


@pytest.fixture(scope="session")
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip("golden file missing — run 14_golden_reference.ipynb first")
    return json.loads(GOLDEN_PATH.read_text())


@pytest.fixture(scope="session")
def fur(golden):
    import torch

    from frames.representations import FrameUnembeddingRepresentation

    cfg = golden["config"]
    return FrameUnembeddingRepresentation.from_model_id(
        cfg["model_id"],
        device_map=cfg["device_map"],
        torch_dtype=torch.float16,
    )


@pytest.mark.gpu
def test_topk_guide_matches_golden(golden, fur):
    import torch

    cfg = golden["config"]
    for run in golden["runs"]:
        texts, probe = fur.quick_generate_with_topk_guide(
            golden["prompts"],
            guide=run["guide"],
            min_lemmas_per_synset=cfg["min_lemmas_per_synset"],
            max_token_count=cfg["max_token_count"],
            k=cfg["k"],
            steps=cfg["steps"],
        )
        assert texts == run["texts"], f"generated text mismatch for guide={run['guide']}"
        torch.testing.assert_close(
            probe[..., -1].cpu().float(),
            torch.tensor(run["probe_final"], dtype=torch.float32),
            rtol=1e-3,
            atol=1e-3,
        )
