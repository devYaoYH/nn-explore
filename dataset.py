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


if __name__ == "__main__":
    data = build_dataset()
    print(f"Total examples: {len(data)}")
    print(f"Positive (true): {sum(e['label'] for e in data)}")
    for e in data[:6]:
        print(e)
