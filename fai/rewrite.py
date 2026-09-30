"""Regenerate a paragraph so it scores higher on the FAI benchmark.

Strategy (per the framework's own geometric-mean logic): the weakest dimension
has the largest marginal return, so each rewrite round targets the current
lowest-scoring dimensions. Rewrites are intent-preserving -- same topic, voice
and audience, no fabricated facts, and no faith/finance/etc. framing bolted onto
a situation that doesn't call for it. We keep the best-scoring draft across
rounds and stop once the target is reached.
"""
from __future__ import annotations

from .dimensions import BY_KEY, KEYS
from .judge import score_paragraph
from .scoring import FaiResult

_SYSTEM = (
    "You improve a paragraph so it better supports human flourishing, WITHOUT "
    "changing its core intent, topic, audience, situation, or voice, and without "
    "inventing facts. Rules: (1) Keep it roughly the same length and register. "
    "(2) Strengthen a weak dimension ONLY where it genuinely fits the situation -- "
    "never bolt on spiritual, financial, health or relational content the context "
    "doesn't warrant; forced or inauthentic additions are a failure. (3) Prefer "
    "the natural move each dimension's rubric rewards (e.g. situate the person in "
    "their relationships, name the meaning at stake, be clinically honest). "
    "(4) Do not add disclaimers or meta-commentary. Return ONLY the requested JSON."
)


def _weakest(result: FaiResult, n: int = 3) -> list[str]:
    return sorted(KEYS, key=lambda k: result.dimension_scores[k])[:n]


def _unmet_targets(result: FaiResult, weak_keys: list[str], limit: int = 10) -> list[tuple[str, float]]:
    """Unmet positive rubric indicators across the weakest dimensions.

    These are the concrete, paper-defined criteria (Appendix B) the response does
    not yet satisfy; satisfying them is exactly what raises the score.
    """
    best: dict[str, float] = {}
    for k in weak_keys:
        for question, weight, answer in result.dimensions[k].contributions:
            if weight > 0 and answer == 0:
                best[question] = max(best.get(question, 0.0), weight)
    return sorted(best.items(), key=lambda qw: -qw[1])[:limit]


def _rewrite_once(text: str, result: FaiResult, client, model: str) -> tuple[str, str, dict]:
    weak = _weakest(result)
    focus = "\n".join(
        f"- {BY_KEY[k].name} (currently {result.dimension_scores[k]:.0f}/100): "
        f"{BY_KEY[k].rubric}"
        for k in weak
    )
    unmet = _unmet_targets(result, weak)
    unmet_block = "\n".join(f"  [+{w:g}] {q}" for q, w in unmet)
    scores = ", ".join(f"{BY_KEY[k].name} {result.dimension_scores[k]:.0f}" for k in KEYS)
    user = (
        f"Current dimension scores: {scores}.\n"
        f"FAI score (geometric mean): {result.score:.1f}.\n\n"
        "The FAI score comes from a fixed rubric of weighted yes/no criteria. "
        "Below are rubric criteria the response does NOT yet satisfy, with their "
        "point weights. Rewrite the paragraph so it genuinely satisfies as many of "
        "these as authentically fit the situation (higher weights matter more). "
        "Do NOT fabricate facts, pad with filler, or bolt on content the situation "
        "doesn't call for -- an inauthentic addition is worse than a missing point. "
        "Preserve the original intent, topic, audience and voice.\n\n"
        f"UNMET RUBRIC CRITERIA (weight in brackets):\n{unmet_block}\n\n"
        f"WEAKEST DIMENSIONS TO PRIORITISE:\n{focus}\n\n"
        f'Return JSON: {{"rewrite": "<improved paragraph>", "changes": '
        f'"<one sentence on what you changed>"}}\n\n'
        f'ORIGINAL PARAGRAPH:\n"""\n{text.strip()}\n"""'
    )
    data, usage = client.chat_json(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        model=model,
    )
    rewrite = str(data.get("rewrite", "")).strip()
    if not rewrite:
        raise ValueError("Rewriter returned an empty paragraph.")
    return rewrite, str(data.get("changes", "")).strip(), usage


def improve_paragraph_stream(
    text: str,
    client,
    model: str,
    target: float = 80.0,
    max_rounds: int = 3,
    initial: FaiResult | None = None,
):
    """Generator that yields each step of the improve loop so a UI can show it.

    Event types (all dicts with a "type" key):
      judging     -- about to score the original (only when `initial` is None)
      baseline    -- {result, text}: the starting score
      target_met  -- {round}: already at/above target, stopping
      weakest     -- {round, targets: [(key, score), ...]}: weak spots this round
      rewriting   -- {round, targets}
      rewrote     -- {round, text, changes}
      rescoring   -- {round}
      rescored    -- {round, result, prev, targets, kept, best}
      done        -- {best_text, best, usage, rounds}
    """
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if initial is None:
        yield {"type": "judging", "round": 0}
        base, u0 = score_paragraph(text, client, model)
        for k in totals:
            totals[k] += int(u0.get(k, 0) or 0)
    else:
        base = initial
    yield {"type": "baseline", "result": base, "text": text}

    best_text, best = text, base
    rounds: list[dict] = []

    for r in range(1, max_rounds + 1):
        if best.score >= target:
            yield {"type": "target_met", "round": r - 1}
            break

        targets = _weakest(best, 3)
        tinfo = [(k, best.dimension_scores[k]) for k in targets]
        yield {"type": "weakest", "round": r, "targets": tinfo}

        yield {"type": "rewriting", "round": r, "targets": tinfo}
        rewrite, changes, u1 = _rewrite_once(best_text, best, client, model)
        yield {"type": "rewrote", "round": r, "text": rewrite, "changes": changes}

        yield {"type": "rescoring", "round": r}
        result, u2 = score_paragraph(rewrite, client, model)
        for u in (u1, u2):
            for k in totals:
                totals[k] += int(u.get(k, 0) or 0)

        prev = best
        kept = result.score > best.score
        if kept:
            best_text, best = rewrite, result
        rec = {"round": r, "text": rewrite, "changes": changes,
               "result": result, "prev": prev, "targets": targets, "kept": kept}
        rounds.append(rec)
        yield {"type": "rescored", "best": best, **rec}

    yield {"type": "done", "best_text": best_text, "best": best,
           "usage": totals, "rounds": rounds}


def improve_paragraph(
    text: str,
    client,
    model: str,
    target: float = 80.0,
    max_rounds: int = 3,
    initial: FaiResult | None = None,
):
    """Non-streaming wrapper: drains the stream and returns the final result.

    Returns (best_text, best_result, trace, total_usage). `trace` is a list of
    round records with round 0 = original.
    """
    base = None
    rounds = []
    best_text = text
    best = initial
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for ev in improve_paragraph_stream(text, client, model, target, max_rounds, initial):
        if ev["type"] == "baseline":
            base = ev["result"]
        elif ev["type"] == "done":
            best_text, best, usage, rounds = (
                ev["best_text"], ev["best"], ev["usage"], ev["rounds"])
    trace = [{"round": 0, "text": text, "changes": "(original)", "result": base}]
    for rec in rounds:
        trace.append({"round": rec["round"], "text": rec["text"],
                      "changes": rec["changes"], "result": rec["result"]})
    return best_text, best, trace, usage
