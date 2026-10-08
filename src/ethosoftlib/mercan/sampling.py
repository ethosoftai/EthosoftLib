"""Python sampler matching Mercan CLI's top-k, top-p and repetition policy."""
from __future__ import annotations

from collections import Counter
import heapq
import math
import random
from typing import Sequence


def sample_token(
    logits: Sequence[float],
    vocab_size: int, *,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repeat_penalty: float = 1.15,
    recent_tokens: Sequence[int] = (),
    rng: random.Random | None = None,
) -> int:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and >= 0")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if not math.isfinite(repeat_penalty) or repeat_penalty <= 0:
        raise ValueError("repeat_penalty must be finite and > 0")

    counts = Counter(int(t) for t in recent_tokens if 0 <= t < vocab_size)

    def adjusted(i: int) -> float:
        value = float(logits[i])
        if math.isnan(value):
            return float("-inf")
        if counts[i] and repeat_penalty != 1.0:
            # Mercan CLI applies the penalty once per occurrence.
            factor = repeat_penalty ** counts[i]
            value = value / factor if value > 0 else value * factor
        return value

    if temperature == 0:
        return max(range(vocab_size), key=adjusted)

    ranked = heapq.nlargest(
        min(top_k, vocab_size),
        ((adjusted(i), i) for i in range(vocab_size)),
        key=lambda pair: pair[0],
    )
    best = ranked[0][0]
    if best == float("-inf"):
        raise ValueError("All logits are invalid or negative infinity")
    if best == float("inf"):
        candidates = [index for value, index in ranked if value == best]
        return (rng or random).choice(candidates)
    weights = [math.exp((value - best) / temperature) for value, _ in ranked]
    total = sum(weights)
    if top_p < 1.0:
        mass = 0.0
        keep = 0
        for weight in weights:
            mass += weight / total
            keep += 1
            if mass >= top_p:
                break
        ranked = ranked[:keep]
        weights = weights[:keep]
    return (rng or random).choices([i for _, i in ranked], weights=weights, k=1)[0]
