"""CPU unit tests for the evaluation harness (fake model, no GPU)."""

import json

import pandas as pd
import torch

from frames.evaluation.harness import EvaluationHarness, RecordingScorer


class FakeScorer:
    """Records protocol calls; sums the concept dimension."""

    def __init__(self):
        self.reset_called_with = None
        self.observe_calls = 0

    def reset(self, n):
        self.reset_called_with = n

    def observe(self, tokens, n):
        self.observe_calls += 1

    def __call__(self, projections, weights):
        return (projections * weights).sum(-1)


class TestRecordingScorer:
    def test_delegates_protocol_and_scoring(self):
        inner = FakeScorer()
        recorder = RecordingScorer(inner)
        recorder.reset(n=2)
        recorder.observe(torch.zeros(4, 3, dtype=torch.long), n=2)

        assert inner.reset_called_with == 2
        assert inner.observe_calls == 1

        proj = torch.randn(4, 5, 1, 2)  # n=2, m=2
        out = recorder(proj, torch.ones(2))
        assert torch.allclose(out, proj.sum(-1))

    def test_records_representative_rows_per_step(self):
        recorder = RecordingScorer(FakeScorer())
        recorder.reset(n=2)
        proj = torch.arange(4 * 3 * 1 * 2, dtype=torch.float).reshape(4, 3, 1, 2)
        recorder(proj, torch.ones(2))
        recorder(proj, torch.ones(2))

        assert len(recorder.steps) == 2
        step = recorder.steps[0]
        # representative rows are 0 and 2 (i * m with m=2)
        assert step["per_concept"][0] == proj[0, -1, 0, :].tolist()
        assert step["per_concept"][1] == proj[2, -1, 0, :].tolist()

        trace = recorder.per_input_steps(1)
        assert len(trace) == 2
        assert trace[0]["per_concept"] == proj[2, -1, 0, :].tolist()

    def test_reset_clears_steps(self):
        recorder = RecordingScorer(FakeScorer())
        recorder.reset(n=1)
        recorder(torch.randn(2, 3, 1, 2), torch.ones(2))
        recorder.reset(n=1)
        assert recorder.steps == []


class FakeTokenizer:
    pass


class FakeData:
    def __init__(self, frame):
        self.frame = frame

    def get_dataframe(self, tokenizer, *args):
        return self.frame


class FakeModel:
    tokenizer = FakeTokenizer()


class FakeFur:
    """Just enough surface for EvaluationHarness."""

    def __init__(self):
        self.model = FakeModel()
        self.data = FakeData(
            pd.DataFrame(
                {
                    "synset": ["joy.n.01", "joy.n.01", "dog.n.01"],
                    "lemma": [" joy", "gladness", " dog"],
                }
            )
        )

    def loss(self, texts):
        # longer text -> higher loss, so ratios are controllable in tests
        return torch.tensor([0.001 * len(texts[0])])

    def generate(self, prompts, **kwargs):
        return [p + " baseline continuation" for p in prompts]


def make_harness(tmp_path, threshold=2.5):
    return EvaluationHarness(
        FakeFur(),
        log_path=tmp_path / "log.jsonl",
        min_lemmas_per_synset=1,
        max_token_count=3,
        ppl_ratio_threshold=threshold,
    )


class TestContinuation:
    def test_exact_prefix_strip(self):
        cont = EvaluationHarness.continuation("PROMPT and more", "PROMPT")
        assert cont == " and more"

    def test_assistant_marker_split(self):
        text = "user stuff assistant<|end_header_id|>the reply"
        assert EvaluationHarness.continuation(text, "unrelated") == "the reply"

    def test_common_prefix_fallback(self):
        # decoded text drops special tokens, so exact prefix match fails
        cont = EvaluationHarness.continuation("hello world tail", "hello world!")
        assert cont == " tail"


class TestSuccess:
    def test_presence_in_continuation_only(self, tmp_path):
        harness = make_harness(tmp_path)
        assert harness.concept_success("full of gladness today", "joy.n.01")
        assert not harness.concept_success("nothing relevant here", "joy.n.01")

    def test_lemma_whitespace_normalized(self, tmp_path):
        harness = make_harness(tmp_path)
        # lemma stored as " joy" must still match
        assert harness.concept_success("pure JOY!", "joy.n.01")


class TestEvaluate:
    def test_jsonl_roundtrip_and_fields(self, tmp_path):
        harness = make_harness(tmp_path)
        records = harness.evaluate(
            prompts=["Tell me PROMPT"],
            texts=["Tell me PROMPT about a dog and joy"],
            synsets=["joy.n.01", "dog.n.01"],
            config={"method": "test", "k": 3},
            probe=torch.tensor([[0.1, 0.5]]),
        )

        assert records[0]["success"] == {"joy.n.01": True, "dog.n.01": True}
        assert records[0]["probe_final"] == 0.5
        assert records[0]["config"]["method"] == "test"

        lines = (tmp_path / "log.jsonl").read_text().strip().split("\n")
        parsed = [json.loads(line) for line in lines]
        assert parsed[0]["success"]["dog.n.01"] is True

    def test_fluency_flag_uses_threshold(self, tmp_path):
        harness = make_harness(tmp_path, threshold=1.01)
        # steered text much longer than baseline -> ppl ratio > threshold
        records = harness.evaluate(
            prompts=["p"],
            texts=["p" + " long steered continuation" * 20],
            synsets=[],
            config={},
        )
        assert records[0]["fluency_flag"] is True

    def test_appends_across_calls(self, tmp_path):
        harness = make_harness(tmp_path)
        for _ in range(2):
            harness.evaluate(["p"], ["p x"], [], config={})
        lines = (tmp_path / "log.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
