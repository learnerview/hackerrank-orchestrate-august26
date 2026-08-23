"""Composition root for the LLM stage.

Builds a fully-wired GroqRoutingLLM from Settings: loads solved examples
from dataset/sample_messages.csv, selects a diverse few-shot set, and wires
the disk cache. Mirrors media/factory.py — the only place that constructs
clients from config.
"""

from __future__ import annotations

from pathlib import Path

from config import Settings
from llm.cache import LLMJsonCache
from llm.groq_client import GroqRoutingLLM
from llm.prompt_builder import FewShotExample, select_few_shot_examples
from models.message import SampleMessage


def load_sample_examples(dataset_dir: Path, n: int = 6) -> list[FewShotExample]:
    """Load and diversity-select few-shot examples from sample_messages.csv.

    Rows are parsed into SampleMessage objects (the same loader used for
    evaluation) so the prompt uses exactly the labels the dataset defines.
    """
    from ingestion import load_sample_messages

    samples: list[SampleMessage] = load_sample_messages(dataset_dir / "sample_messages.csv")
    return select_few_shot_examples(samples, n=n)


def build_routing_llm(
    settings: Settings,
    few_shot_examples: list[FewShotExample] | None = None,
) -> GroqRoutingLLM:
    if few_shot_examples is None:
        few_shot_examples = load_sample_examples(settings.dataset_dir)
    return GroqRoutingLLM(
        settings=settings,
        few_shot_examples=few_shot_examples,
        cache=LLMJsonCache(settings.llm_cache_dir),
    )
