import re
from collections import defaultdict

YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def _split_year(label: str):
    tokens = label.strip().split()
    if tokens and YEAR_RE.match(tokens[-1]):
        return " ".join(tokens[:-1]), int(tokens[-1])
    return label, None


def build_make_map_from_dataset(class_names):
    """
    Build a make map that matches THIS dataset's naming pattern.

    Dataset-derived 2-word makes are:
    - Aston Martin (Aston always followed by Martin, appears multiple times)
    - Land Rover (Land always followed by Rover, appears multiple times)
    - AM General (AM always followed by General, appears once but should still be 2-word)

    We detect:
    1) first word appears >= 2 AND always has the same second word => 2-word make
    2) special case for AM General (first token 'AM' + second 'General')
    """

    first_counts = defaultdict(int)
    first_to_seconds = defaultdict(set)
    first_to_second_counts = defaultdict(lambda: defaultdict(int))

    for name in class_names:
        base, _ = _split_year(name)
        words = base.split()
        if not words:
            continue

        first_counts[words[0]] += 1

        if len(words) >= 2:
            first_to_seconds[words[0]].add(words[1])
            first_to_second_counts[words[0]][words[1]] += 1

    make_map = {}

    for first, count in first_counts.items():
        seconds = first_to_seconds.get(first, set())

        # Special case: AM General (appears once, but is clearly a 2-word make here)
        if first == "AM" and "General" in seconds:
            make_map[first] = (2, "AM General")
            continue

        # General rule: if first appears multiple times AND second word is always the same => 2-word make
        if count >= 2 and len(seconds) == 1:
            second = next(iter(seconds))
            make_map[first] = (2, f"{first} {second}")
        else:
            make_map[first] = (1, first)

    return make_map


def parse_predicted_class(predicted_class: str, make_map):
    """
    Parse predicted_class into:
    - brand (make): 1 or 2 words based on make_map
    - model: everything after make, before year
    - year: last token if it is a 4-digit year
    """
    base, year = _split_year(predicted_class)
    words = base.split()

    if not words:
        return {"brand": None, "model": "", "year": year}

    first = words[0]
    make_len, brand = make_map.get(first, (1, first))

    model = " ".join(words[make_len:]).strip()
    return {"brand": brand, "model": model, "year": year}
