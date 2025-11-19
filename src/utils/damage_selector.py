def compute_damage_score(detections):

    if not detections:
        return 0.0

    # Simple but effective scoring:
    # Sum of all confidence values (more damage → more detections → higher score)
    score = sum(det["confidence"] for det in detections if "confidence" in det)

    return round(float(score), 4)


def select_top_k_images(predictions, k=2):

    # Attach score
    for item in predictions:
        detections = item.get("detections", [])
        item["damage_score"] = compute_damage_score(detections)

    # Sort descending by score
    predictions_sorted = sorted(predictions, key=lambda x: x["damage_score"], reverse=True)

    return predictions_sorted[:k]


def process_damage_outputs(damage_outputs, k=2):

    if damage_outputs is None or len(damage_outputs) == 0:
        return []

    top_k = select_top_k_images(damage_outputs, k=k)

    return top_k
