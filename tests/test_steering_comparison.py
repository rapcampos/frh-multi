"""CPU integration contracts for the matched comparison experiment."""

import copy
import json

import pytest
import torch

from frames.experiments.steering_comparison import (
    ModelBackend,
    clean_continuation,
    custom_token_rows,
    frame_guide,
    make_experiment,
    matrix_size,
    resolve_synsets,
    run_battery,
)
from frames.representations.concept import Concept


class FakeBackend:
    provenance = {"test_backend": True}

    def __init__(self, fail_measure=None):
        self.baseline_calls = 0
        self.generate_calls = 0
        self.measure_calls = 0
        self.fail_measure = fail_measure

    def baseline(self, prompts):
        self.baseline_calls += 1
        return ["baseline:" + p["id"] for p in prompts]

    def generate(self, config, source, method, lam, prompts):
        self.generate_calls += 1
        return ["steered:" + p["id"] for p in prompts]

    def measure(self, text, baseline, config):
        self.measure_calls += 1
        if self.fail_measure == self.measure_calls:
            raise RuntimeError("interrupted")
        assert text.split(":")[1] == baseline.split(":")[1]
        return {"valid": True}


def small_manifest():
    manifest = make_experiment(smoke=True)
    manifest["configs"] = manifest["configs"][:2]
    return manifest


def test_full_and_smoke_matrix_are_complete():
    manifest = make_experiment()
    assert len(manifest["configs"]) == 74
    assert matrix_size(manifest) == 4736
    assert (
        manifest["settings"]["baseline_max_new_tokens"]
        == manifest["settings"]["steps"] + 1
    )
    smoke = make_experiment(smoke=True)
    assert matrix_size(smoke) == 192
    for count in range(1, 5):
        configs = [c for c in smoke["configs"] if c["unit_count"] == count]
        assert len(configs) == 3
        assert sum(bool(c["negative"]) for c in configs) == 2


def test_run_pairs_baselines_and_resumes_without_regeneration(tmp_path):
    manifest = small_manifest()
    backend = FakeBackend()
    prefix = tmp_path / "comparison"
    records = run_battery(manifest, backend, prefix)
    assert len(records) == matrix_size(manifest)
    assert backend.baseline_calls == 1
    calls = backend.generate_calls
    assert run_battery(manifest, backend, prefix) == records
    assert backend.baseline_calls == 1
    assert backend.generate_calls == calls


def test_partial_batch_resume_preserves_original_prompt_batch(tmp_path):
    manifest = small_manifest()
    prefix = tmp_path / "comparison"
    with pytest.raises(RuntimeError, match="interrupted"):
        run_battery(manifest, FakeBackend(fail_measure=2), prefix)
    saved = [
        json.loads(line)
        for line in prefix.with_suffix(".jsonl").read_text().splitlines()
    ]
    assert len(saved) == 1
    backend = FakeBackend()
    records = run_battery(manifest, backend, prefix)
    assert len(records) == matrix_size(manifest)
    assert records[0] == saved[0]
    assert len({r["run_id"] for r in records}) == len(records)
    assert backend.baseline_calls == 0


@pytest.mark.parametrize("change", ["lambda", "word", "prompt", "construction"])
def test_resume_rejects_changed_identity(tmp_path, change):
    manifest = small_manifest()
    backend = FakeBackend()
    prefix = tmp_path / "comparison"
    run_battery(manifest, backend, prefix)
    changed = copy.deepcopy(manifest)
    if change == "lambda":
        changed["settings"]["lambdas"] = [0.0, 0.5]
    elif change == "word":
        changed["concepts"][0]["words"][0] = "changed"
    elif change == "prompt":
        changed["prompts"][0]["text"] = "changed"
    else:
        backend.provenance = {"changed": True}
    with pytest.raises(ValueError, match="Incompatible"):
        run_battery(changed, backend, prefix)


def test_resume_rejects_modified_baseline(tmp_path):
    manifest = small_manifest()
    prefix = tmp_path / "comparison"
    run_battery(manifest, FakeBackend(), prefix)
    path = tmp_path / "comparison_baselines.json"
    data = json.loads(path.read_text())
    data["continuations"][0] = "modified"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Baseline content changed"):
        run_battery(manifest, FakeBackend(), prefix)


def test_continuation_excludes_prompt_and_assistant_prefix():
    prompt = {"prefix": "Once"}
    decoded = "user<|end_header_id|>Tell a story.assistant<|end_header_id|>Once upon a time<|eot_id|>"
    assert clean_continuation(decoded, prompt, ["<|eot_id|>"]) == "upon a time"
    with pytest.raises(ValueError, match="assistant boundary"):
        clean_continuation("not a decoded conversation", prompt)
    with pytest.raises(ValueError, match="assistant prefix"):
        clean_continuation("assistant<|end_header_id|>Wrong", prompt)


def test_custom_token_rows_share_padding_and_record_exclusions():
    class Tokenizer:
        bos_token_id = 1
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            return {"input_ids": [2] * (5 if text.strip() == "long" else 2)}

    rows, excluded = custom_token_rows(Tokenizer(), ["short", "long"], 4)
    assert rows == [[2, 2, 0, 0]] * 4
    assert excluded == ["long"]


def test_wordnet_aliases_resolve_without_changing_requested_ids():
    from types import SimpleNamespace

    concepts = [
        {"concept": "trust", "synset": "trust.n.05"},
        {"concept": "suspicion", "synset": "suspicion.n.02"},
    ]
    aliases = {"trust.n.05": "faith.n.02", "suspicion.n.02": "misgiving.n.03"}
    corpus = SimpleNamespace(
        synset=lambda requested: SimpleNamespace(name=lambda: aliases[requested])
    )
    assert resolve_synsets(concepts, corpus) == {
        "trust": "faith.n.02",
        "suspicion": "misgiving.n.03",
    }
    assert concepts[0]["synset"] == "trust.n.05"


def test_frame_counterfactual_is_an_equal_weight_unit():
    concepts = {
        name: Concept(synset=name, tensor=torch.eye(3)[:, i].reshape(1, 3, 1))
        for i, name in enumerate(["a", "b", "c"])
    }
    guide = frame_guide({"positive": ["a", "c"], "negative": ["b"]}, concepts)
    # The inner difference has unit norm; then it receives the same weight as c.
    expected = torch.tensor([0.5, -0.5, 2**-0.5]).reshape(1, 3, 1)
    assert torch.allclose(guide.tensor, expected, atol=1e-6)
    single = frame_guide({"positive": ["a"], "negative": []}, concepts)
    assert single is concepts["a"]


def test_frh_backend_passes_one_precomposed_guide_without_score_normalization():
    from types import SimpleNamespace

    backend = object.__new__(ModelBackend)
    backend.settings = {"k": 4, "steps": 24}
    concepts = {
        name: Concept(synset=name, tensor=torch.eye(2)[:, i].reshape(1, 2, 1))
        for i, name in enumerate(["a", "b"])
    }
    backend.frames = {"custom": concepts}
    captured = {}

    def generate(inputs, **kwargs):
        captured.update(kwargs)
        return ["assistant<|end_header_id|>Once result"], None

    backend.fur = SimpleNamespace(
        generate_with_topk_multi_guide=generate,
        model=SimpleNamespace(tokenizer=SimpleNamespace(all_special_tokens=[])),
    )
    result = backend.generate(
        {"positive": ["a", "b"], "negative": []},
        "custom",
        "frh_blend",
        0.4,
        [{"text": "prompt", "prefix": "Once"}],
    )
    assert result == ["result"]
    assert len(captured["guides"]) == 1
    assert captured["normalize"] is None
    assert captured["fluency_weight"] == 0.4
    assert torch.allclose(captured["guides"][0].tensor, torch.full((1, 2, 1), 2**-0.5))


def test_real_metrics_flow_through_runner_and_report(tmp_path):
    from types import SimpleNamespace

    from frames.experiments.comparison_metrics import evaluate_continuation, ppl_metrics
    from frames.experiments.comparison_report import write_report

    manifest = small_manifest()
    model = SimpleNamespace(
        tokenizer=SimpleNamespace(all_special_ids=[0]),
        tokenize=lambda text: torch.tensor(
            [[0, 1, 1 if text.startswith("baseline") else 2]]
        ),
        forward_last_hiden_state=lambda tokens: torch.nn.functional.one_hot(
            tokens.clamp_max(1), 2
        ).float(),
    )
    fur = SimpleNamespace(model=model)
    references = {
        view: {entry["concept"]: torch.eye(2)[:, :1] for entry in manifest["concepts"]}
        for view in (
            "wordnet_frame",
            "wordnet_subspace",
            "custom_frame",
            "custom_subspace",
        )
    }

    class MetricBackend(FakeBackend):
        def measure(self, text, baseline, config):
            return {
                "cosine": evaluate_continuation(
                    fur,
                    text,
                    baseline,
                    references,
                    config["positive"],
                    config["negative"],
                ),
                "fluency": ppl_metrics(12.0, 10.0),
                "continuation_tokens": 2,
                "baseline_tokens": 2,
            }

    prefix = tmp_path / "comparison"
    records = run_battery(manifest, MetricBackend(), prefix)
    paths = write_report(prefix)
    status = json.loads(paths["report_status"].read_text())
    assert status["status"] == "complete"
    assert status["recorded_runs"] == len(records)
    assert status["cosine_invalid_view_rows"] == 0
    assert paths["cosine_ppl"].stat().st_size > 1000
