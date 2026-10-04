"""CPU unit tests for the shared notebook helpers."""

import pandas as pd
import pytest

from frames.experiments import notebook_helpers as nh


class TestChat:
    def test_llama_round_trip(self):
        text = nh.llama_chat("What men can be?", "1.")
        assert nh.answer(text + " Kind") == "1. Kind"

    def test_answer_passthrough_without_marker(self):
        assert nh.answer("plain text") == "plain text"

    def test_chat_fn_dispatch(self):
        assert nh.chat_fn("meta-llama/Llama-3.1-8B-Instruct") is nh.llama_chat
        assert nh.chat_fn("google/gemma-2-9b-it") is nh.gemma_chat
        with pytest.raises(ValueError):
            nh.chat_fn("microsoft/phi-3")


class TestReadJsonl:
    def test_parses_records(self, tmp_path):
        log = tmp_path / "log.jsonl"
        log.write_text('{"a": 1}\n{"a": 2}\n')
        assert nh.read_jsonl(log) == [{"a": 1}, {"a": 2}]


class TestParetoFront:
    def test_dominated_point_removed(self):
        points = pd.DataFrame(
            {
                "ppl_ratio": [1.0, 2.0, 3.0],
                "joint_z": [0.1, 0.5, 0.4],  # third row dominated by second
            }
        )
        front = nh.pareto_front(points)
        assert list(front.index) == [0, 1]

    def test_front_sorted_by_cost(self):
        points = pd.DataFrame({"ppl_ratio": [3.0, 1.0], "joint_z": [0.9, 0.1]})
        front = nh.pareto_front(points)
        assert list(front["ppl_ratio"]) == [1.0, 3.0]

    def test_custom_columns(self):
        points = pd.DataFrame({"cost": [1.0, 2.0], "gain": [0.5, 0.4]})
        front = nh.pareto_front(points, cost="cost", gain="gain")
        assert list(front.index) == [0]

    def test_ties_are_kept(self):
        points = pd.DataFrame({"ppl_ratio": [1.0, 1.0], "joint_z": [0.5, 0.5]})
        assert len(nh.pareto_front(points)) == 2
