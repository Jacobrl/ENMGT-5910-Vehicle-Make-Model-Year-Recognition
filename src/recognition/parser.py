import re

def parse_top1_prediction(output_text):
    """
    Parse model's printed output and extract:
    - brand
    - model
    - year
    - confidence (0 to 1)
    """

    # Find the first prediction line
    lines = output_text.strip().split("\n")
    top_line = None
    for line in lines:
        if line.strip().startswith("1. "):
            top_line = line.strip()
            break

    if not top_line:
        raise ValueError("Top-1 prediction line not found in output.")

    # Remove "1. " prefix
    top_line = top_line[3:].strip()

    # Extract confidence
    confidence_match = re.search(r"(\d+\.?\d*)%", top_line)
    confidence = float(confidence_match.group(1)) / 100 if confidence_match else None

    # Remove the "— 55.83%" part
    car_text = re.sub(r"[-—–].*", "", top_line).strip()

    # Extract year (last 4 numbers)
    year_match = re.search(r"(\d{4})$", car_text)
    year = int(year_match.group(1)) if year_match else None

    # Remove year from the string
    car_text = car_text.replace(str(year), "").strip()

    # First word = brand
    parts = car_text.split(" ", 1)
    brand = parts[0]

    # Remaining = model name
    model = parts[1] if len(parts) > 1 else ""

    return {
        "brand": brand,
        "model": model,
        "year": year,
        "confidence": confidence
    }
