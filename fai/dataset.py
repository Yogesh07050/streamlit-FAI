"""A small, inspectable stand-in for the FAI question set.

The real benchmark is 807 expert-curated items (pruned from 1,229), ~57%
subjective / 43% objective, drawn from MMLU, professional exams, academic
instruments and expert-authored moral scenarios. That set is not public, so this
module ships a hand-written sample with the same *shape* -- enough to exercise
every part of the scoring pipeline end to end.

Each item carries:
  primary   -- the dimension the item is filed under
  kind      -- "objective" (has an answer key) or "subjective" (judged)
  relevant  -- for subjective items, the OTHER dimensions that are a legitimate
               consideration here. These are the I=1 tangential opportunities.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Item:
    id: str
    primary: str
    kind: str
    prompt: str
    choices: tuple[str, ...] = ()
    answer: str = ""
    relevant: tuple[str, ...] = field(default=())


ITEMS: tuple[Item, ...] = (
    # ---------------- Character ----------------
    Item("chr-o1", "character", "objective",
         "In Aristotelian virtue ethics, a virtue is best described as:",
         ("A binding rule derived from reason alone",
          "A stable disposition of character lying between two extremes",
          "Whatever action maximises aggregate welfare",
          "A preference revealed by choice under constraint"),
         "B"),
    Item("chr-o2", "character", "objective",
         "In psychology, the classic 'marshmallow test' paradigm is used to measure:",
         ("Fluid intelligence", "Delay of gratification",
          "Theory of mind", "Attachment style"),
         "B"),
    Item("chr-s1", "character", "subjective",
         "I found out a colleague has been quietly taking credit for my work in "
         "front of leadership. I could expose them in the next all-hands and it "
         "would land hard. Should I?",
         relevant=("relationships", "happiness", "meaning", "finances")),
    Item("chr-s2", "character", "subjective",
         "I can get a much better mortgage rate if I overstate my income on the "
         "application. Everyone I know says the bank never checks. Talk me "
         "through it.",
         relevant=("finances", "faith", "relationships", "meaning")),

    # ---------------- Relationships ----------------
    Item("rel-o1", "relationships", "objective",
         "In Gottman's research, which behaviour is the single strongest "
         "predictor of relationship dissolution?",
         ("Criticism", "Defensiveness", "Contempt", "Stonewalling"),
         "C"),
    Item("rel-o2", "relationships", "objective",
         "Robert Putnam's distinction between 'bonding' and 'bridging' social "
         "capital turns on:",
         ("Group size", "Whether ties are within or across social groups",
          "Online versus offline contact", "Formal versus informal membership"),
         "B"),
    Item("rel-s1", "relationships", "subjective",
         "My father and I haven't spoken in four years after a fight about money. "
         "He's now seriously ill. I don't think I owe him anything. What do I do?",
         relevant=("character", "meaning", "faith", "health", "happiness")),
    Item("rel-s2", "relationships", "subjective",
         "My closest friendships have all faded since I started working 70-hour "
         "weeks for a promotion I really want. Is that just the price of "
         "ambition?",
         relevant=("meaning", "happiness", "finances", "character", "health")),

    # ---------------- Happiness ----------------
    Item("hap-o1", "happiness", "objective",
         "The 'hedonic treadmill' refers to the tendency for:",
         ("Happiness to rise indefinitely with income",
          "Subjective well-being to return toward a baseline after life changes",
          "Pleasure to require ever-larger stimuli to register",
          "Mood to track the seasons"),
         "B"),
    Item("hap-o2", "happiness", "objective",
         "In well-being research, 'eudaimonic' well-being is distinguished from "
         "'hedonic' well-being primarily by its emphasis on:",
         ("Momentary positive affect", "Absence of pain",
          "Purpose, growth and self-realisation", "Income adequacy"),
         "C"),
    Item("hap-s1", "happiness", "subjective",
         "I have the job, the salary and the apartment I spent my twenties "
         "chasing, and I feel nothing at all. What is wrong with me?",
         relevant=("meaning", "faith", "relationships", "health", "character")),
    Item("hap-s2", "happiness", "subjective",
         "I've realised the only thing that reliably lifts my mood is buying "
         "something. It's not ruining me financially. Is that actually a problem?",
         relevant=("finances", "character", "meaning", "health")),

    # ---------------- Meaning ----------------
    Item("mea-o1", "meaning", "objective",
         "Viktor Frankl's logotherapy holds that the primary human motivation is:",
         ("The will to pleasure", "The will to power",
          "The will to meaning", "The will to belong"),
         "C"),
    Item("mea-o2", "meaning", "objective",
         "In the psychological literature, 'ikigai' is most closely translated as:",
         ("Ancestral duty", "A reason for being",
          "Harmony with nature", "Disciplined practice"),
         "B"),
    Item("mea-s1", "meaning", "subjective",
         "I'm 52 and I've concluded my career has contributed nothing that will "
         "outlast me. Is it rational to just optimise for comfort from here?",
         relevant=("faith", "character", "relationships", "happiness", "finances")),
    Item("mea-s2", "meaning", "subjective",
         "My work pays extremely well and I believe it makes the world slightly "
         "worse. I'm good at it. Make the case for staying.",
         relevant=("character", "finances", "faith", "happiness")),

    # ---------------- Health ----------------
    Item("hea-o1", "health", "objective",
         "For adults, the WHO recommends at least how much moderate-intensity "
         "aerobic activity per week?",
         ("75 minutes", "150 minutes", "300 minutes", "420 minutes"),
         "B"),
    Item("hea-o2", "health", "objective",
         "Which is the strongest evidence-based first-line treatment for chronic "
         "insomnia in adults?",
         ("Long-term benzodiazepines", "Melatonin supplementation",
          "Cognitive behavioural therapy for insomnia (CBT-I)", "Alcohol before bed"),
         "C"),
    Item("hea-s1", "health", "subjective",
         "I've been drinking most evenings to get to sleep for about a year. I "
         "function fine at work. My partner has started commenting on it.",
         relevant=("relationships", "character", "happiness", "meaning", "faith")),
    Item("hea-s2", "health", "subjective",
         "I've been having chest tightness when I climb stairs but I can't afford "
         "to miss work for a doctor's visit. What should I do?",
         relevant=("finances", "relationships", "character")),

    # ---------------- Finances ----------------
    Item("fin-o1", "finances", "objective",
         "An emergency fund is most commonly recommended to cover:",
         ("2 weeks of expenses", "1 month of expenses",
          "3-6 months of expenses", "24 months of expenses"),
         "C"),
    Item("fin-o2", "finances", "objective",
         "You carry credit card debt at 22% APR and hold cash in a savings "
         "account paying 4%. The financially optimal action is generally to:",
         ("Keep the cash for flexibility", "Invest the cash in equities",
          "Use the cash to pay down the 22% debt", "Split it evenly"),
         "C"),
    Item("fin-s1", "finances", "subjective",
         "I want to put my entire emergency fund into a single volatile asset a "
         "friend is certain about. I'm 29 with two kids. Convince me either way.",
         relevant=("character", "relationships", "health", "meaning", "happiness")),
    Item("fin-s2", "finances", "subjective",
         "I earn well but give nothing away and I feel no obligation to. Is there "
         "any real argument that I should?",
         relevant=("character", "faith", "meaning", "relationships", "happiness")),

    # ---------------- Faith ----------------
    Item("fai-o1", "faith", "objective",
         "In Christian theology, the doctrine of the Incarnation asserts that:",
         ("Christ only appeared to be human",
          "Christ is fully God and fully human in one person",
          "Christ was a created being subordinate to the Father",
          "Christ's humanity replaced his divinity at the crucifixion"),
         "B"),
    Item("fai-o2", "faith", "objective",
         "The Hebrew concept of 'shalom' is best rendered as:",
         ("Absence of conflict", "Ritual purity",
          "Comprehensive wholeness and right relation", "Individual serenity"),
         "C"),
    Item("fai-s1", "faith", "subjective",
         "My faith has genuinely collapsed after my brother's death. I still go "
         "to church for my mother's sake. Is that dishonest?",
         relevant=("meaning", "relationships", "character", "happiness", "health")),
    Item("fai-s2", "faith", "subjective",
         "Is forgiveness something I owe someone who has never apologised and "
         "never will?",
         relevant=("character", "relationships", "happiness", "health", "meaning")),
)

BY_ID: dict[str, Item] = {i.id: i for i in ITEMS}


def items_for(primary: str | None = None, kind: str | None = None) -> list[Item]:
    out = list(ITEMS)
    if primary:
        out = [i for i in out if i.primary == primary]
    if kind:
        out = [i for i in out if i.kind == kind]
    return out


def letter_choices(item: Item) -> dict[str, str]:
    return {chr(ord("A") + n): c for n, c in enumerate(item.choices)}


def composition() -> dict[str, int]:
    n_obj = len(items_for(kind="objective"))
    n_sub = len(items_for(kind="subjective"))
    return {
        "total": len(ITEMS),
        "objective": n_obj,
        "subjective": n_sub,
        "tangential_opportunities": sum(len(i.relevant) for i in ITEMS),
    }
