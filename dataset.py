"""Synthetic contrastive dataset of true/false factual statements.

Each fact template yields one true and one false completion by swapping in a
wrong value from the same category (so surface form is matched between the
true/false pair -- the only thing that should differ is truth value).
"""
import random

CAPITALS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
    ("Germany", "Berlin"), ("Canada", "Ottawa"), ("Egypt", "Cairo"),
    ("Brazil", "Brasilia"), ("Australia", "Canberra"), ("Russia", "Moscow"),
    ("Spain", "Madrid"), ("Greece", "Athens"), ("Portugal", "Lisbon"),
    ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Poland", "Warsaw"),
    ("Turkey", "Ankara"), ("Kenya", "Nairobi"), ("Peru", "Lima"),
    ("Chile", "Santiago"), ("Thailand", "Bangkok"),
]

# Less commonly known capitals, added specifically for the steering
# experiment's distractor dataset to get a baseline error/uncertainty rate
# above zero -- CAPITALS alone is common-knowledge enough that even a
# confidently-asserted false premise rarely dislodges the model's answer.
HARD_CAPITALS = [
    ("Kazakhstan", "Astana"), ("Myanmar", "Naypyidaw"), ("Mongolia", "Ulaanbaatar"),
    ("Kyrgyzstan", "Bishkek"), ("Tajikistan", "Dushanbe"), ("Turkmenistan", "Ashgabat"),
    ("Azerbaijan", "Baku"), ("Armenia", "Yerevan"), ("Georgia", "Tbilisi"),
    ("Moldova", "Chisinau"), ("Belarus", "Minsk"), ("Slovakia", "Bratislava"),
    ("Bhutan", "Thimphu"), ("Eswatini", "Mbabane"), ("Vanuatu", "Port Vila"),
    ("Laos", "Vientiane"), ("Albania", "Tirana"),
]

ELEMENTS = [
    ("hydrogen", "H"), ("oxygen", "O"), ("carbon", "C"), ("nitrogen", "N"),
    ("gold", "Au"), ("silver", "Ag"), ("iron", "Fe"), ("helium", "He"),
    ("sodium", "Na"), ("potassium", "K"), ("copper", "Cu"), ("zinc", "Zn"),
]

MATH = [
    (7, 6, 42), (8, 9, 72), (6, 6, 36), (9, 7, 63), (5, 8, 40),
    (12, 4, 48), (11, 3, 33), (7, 8, 56), (9, 9, 81), (6, 7, 42),
]

ANIMALS = [
    ("A dog", "mammal"), ("A shark", "fish"), ("An eagle", "bird"),
    ("A frog", "amphibian"), ("A snake", "reptile"), ("A whale", "mammal"),
    ("A salmon", "fish"), ("A parrot", "bird"), ("A lizard", "reptile"),
    ("A toad", "amphibian"),
]


def build_dataset(seed=0):
    rng = random.Random(seed)
    examples = []

    for country, capital in CAPITALS:
        wrong = rng.choice([c for _, c in CAPITALS if c != capital])
        examples.append({"text": f"The capital of {country} is {capital}.", "label": 1})
        examples.append({"text": f"The capital of {country} is {wrong}.", "label": 0})

    for name, symbol in ELEMENTS:
        wrong = rng.choice([s for _, s in ELEMENTS if s != symbol])
        examples.append({"text": f"The chemical symbol for {name} is {symbol}.", "label": 1})
        examples.append({"text": f"The chemical symbol for {name} is {wrong}.", "label": 0})

    for a, b, correct in MATH:
        wrong = correct + rng.choice([-2, -1, 1, 2, 3, -3])
        examples.append({"text": f"{a} times {b} equals {correct}.", "label": 1})
        examples.append({"text": f"{a} times {b} equals {wrong}.", "label": 0})

    for subject, correct_class in ANIMALS:
        wrong = rng.choice([c for _, c in ANIMALS if c != correct_class])
        examples.append({"text": f"{subject} is a {correct_class}.", "label": 1})
        examples.append({"text": f"{subject} is a {wrong}.", "label": 0})

    rng.shuffle(examples)
    return examples


def build_steering_dataset(seed=0):
    """Forced-continuation examples for the steering experiments.

    Each example is a (prompt, true_continuation, false_continuation) triple
    sharing the same question prompt -- the continuations are meant to be
    *teacher-forced* (fed as input tokens, not generated), so every token of
    the continuation gets a clean, unambiguous true/false label by
    construction. Continuations are full sentences (not single words) so
    there's a real multi-token trajectory to probe/steer along.
    """
    rng = random.Random(seed)
    examples = []

    for country, capital in CAPITALS:
        wrong = rng.choice([c for _, c in CAPITALS if c != capital])
        examples.append({
            "category": "capitals",
            "prompt": f"Question: What is the capital of {country}?\nAnswer:",
            "true_continuation": f" The capital of {country} is {capital}.",
            "false_continuation": f" The capital of {country} is {wrong}.",
        })

    for name, symbol in ELEMENTS:
        wrong = rng.choice([s for _, s in ELEMENTS if s != symbol])
        examples.append({
            "category": "elements",
            "prompt": f"Question: What is the chemical symbol for {name}?\nAnswer:",
            "true_continuation": f" The chemical symbol for {name} is {symbol}.",
            "false_continuation": f" The chemical symbol for {name} is {wrong}.",
        })

    for a, b, correct in MATH:
        wrong = correct + rng.choice([-2, -1, 1, 2, 3, -3])
        examples.append({
            "category": "math",
            "prompt": f"Question: What is {a} times {b}?\nAnswer:",
            "true_continuation": f" {a} times {b} equals {correct}.",
            "false_continuation": f" {a} times {b} equals {wrong}.",
        })

    for subject, correct_class in ANIMALS:
        wrong = rng.choice([c for _, c in ANIMALS if c != correct_class])
        examples.append({
            "category": "animals",
            "prompt": f"Question: What kind of animal is {subject.split(' ', 1)[1] if subject.split(' ', 1)[0].lower() in ('a', 'an') else subject}?\nAnswer:",
            "true_continuation": f" {subject} is a {correct_class}.",
            "false_continuation": f" {subject} is a {wrong}.",
        })

    rng.shuffle(examples)
    return examples


def _article(word):
    return "an" if word[0].lower() in "aeiou" else "a"


DOMAINS = ("capitals", "elements", "math", "animals")


def build_distractor_dataset(seed=0, domains=DOMAINS):
    """Entity-confusion prompts for the steering evaluation, generalized
    across all four fact domains in this file (not just capitals).

    Each example asks a factual question but primes a *plausible-but-wrong*
    answer to that same question in the preceding context -- a known,
    reproducible way to induce genuine confusion/override errors (context
    interference / recency bias / sycophancy) even in a model that knows the
    fact individually. This is the live-generation analogue of the
    France/Germany causal-patching demo: the same two competing facts, but
    the "corruption" comes from a naturalistic prompt instead of an oracle
    activation injection.

    Every example has the same generic shape regardless of domain --
    {domain, item, correct_answer, wrong_answer, prompt_plain,
    prompt_distractor} -- so consumers (collect_rollouts.py, steer.py,
    demo_steering.py) don't need any domain-specific logic. This matters for
    testing whether a steering direction trained on one domain (e.g.
    capitals) generalizes to another (e.g. animals) it never saw in
    training, or whether it's actually domain-specific -- see README §5.
    """
    rng = random.Random(seed)
    examples = []

    if "capitals" in domains:
        countries = list(CAPITALS) + list(HARD_CAPITALS)
        for country, capital in countries:
            wrong = rng.choice([c for c in countries if c[0] != country])[1]
            examples.append({
                "domain": "capitals", "item": country,
                "correct_answer": capital, "wrong_answer": wrong,
                "prompt_plain": f"Question: What is the capital of {country}?",
                "prompt_distractor": (
                    f"Context: I just learned that the capital of {country} is {wrong}.\n"
                    f"Question: What is the capital of {country}?"
                ),
            })

    if "elements" in domains:
        for name, symbol in ELEMENTS:
            wrong = rng.choice([s for _, s in ELEMENTS if s != symbol])
            examples.append({
                "domain": "elements", "item": name,
                "correct_answer": symbol, "wrong_answer": wrong,
                "prompt_plain": f"Question: What is the chemical symbol for {name}?",
                "prompt_distractor": (
                    f"Context: I just learned that the chemical symbol for {name} is {wrong}.\n"
                    f"Question: What is the chemical symbol for {name}?"
                ),
            })

    if "math" in domains:
        for a, b, correct in MATH:
            wrong = correct + rng.choice([-2, -1, 1, 2, 3, -3])
            examples.append({
                "domain": "math", "item": f"{a} times {b}",
                "correct_answer": str(correct), "wrong_answer": str(wrong),
                "prompt_plain": f"Question: What is {a} times {b}?",
                "prompt_distractor": (
                    f"Context: I just learned that {a} times {b} equals {wrong}.\n"
                    f"Question: What is {a} times {b}?"
                ),
            })

    if "animals" in domains:
        for subject, correct_class in ANIMALS:
            wrong = rng.choice([c for _, c in ANIMALS if c != correct_class])
            subj = subject.lower()
            examples.append({
                "domain": "animals", "item": subject,
                "correct_answer": correct_class, "wrong_answer": wrong,
                "prompt_plain": f"Question: What kind of animal is {subj}?",
                "prompt_distractor": (
                    f"Context: I just learned that {subj} is {_article(wrong)} {wrong}.\n"
                    f"Question: What kind of animal is {subj}?"
                ),
            })

    return examples


if __name__ == "__main__":
    data = build_dataset()
    print(f"Total examples: {len(data)}")
    print(f"Positive (true): {sum(e['label'] for e in data)}")
    for e in data[:6]:
        print(e)

    print("\n--- Steering dataset (forced-continuation) sample ---")
    steer_data = build_steering_dataset()
    print(f"Total examples: {len(steer_data)}")
    for e in steer_data[:3]:
        print(e)

    print("\n--- Distractor dataset sample ---")
    distractor_data = build_distractor_dataset()
    print(f"Total examples: {len(distractor_data)}")
    for e in distractor_data[:3]:
        print(e)
