from PIL import Image
from qwen_vl_utils import process_vision_info

import torch
from transformers import AutoTokenizer, AutoModelForVision2Seq

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

_tokenizer = None
_model = None


def get_qwen():
    """Lazy-load tokenizer + model once."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print("Loading Qwen2-VL-7B-Instruct...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        _model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True
        )
        print("Qwen2-VL loaded successfully.")
    return _tokenizer, _model


def build_prompt(metadata):
    car = metadata["car"]
    damage = metadata["damage"]
    cost = metadata["cost_estimate"]

    damage_lines = []
    for d in damage:
        classes = [x["class"] for x in d.get("detections", [])]
        classes_text = ", ".join(classes) if classes else "No detected damage"
        damage_lines.append(f"- {classes_text} (score {d['damage_score']:.2f})")

    damage_summary = "\n".join(damage_lines) if damage_lines else "No visible damage."

    usd = cost.get("usd")
    mxn = cost.get("mxn")

    usd_text = f"${usd}" if usd is not None else "not yet estimated"
    mxn_text = f"${mxn} MXN" if mxn is not None else "not yet estimated"

    prompt = f"""
You are an expert automotive claims adjuster.

Use the car info, damage summary, and cost estimate below to write
two SHORT paragraph-style summaries of the situation.

CAR INFO:
- Brand: {car['brand']}
- Model: {car['model']}
- Year: {car['year']}
- Recognition confidence: {car['confidence']:.2f}

DAMAGE SUMMARY:
{damage_summary}

COST ESTIMATE (internal data):
- USD: {usd_text}
- MXN: {mxn_text}

TASK:
1. First write ONE short paragraph in English that:
   - Mentions the car's brand, model, and year.
   - Describes the main visible damage.
   - Mentions the repair cost in USD and MXN
     (say that it is {usd_text} and {mxn_text} if not fully known).

2. Then write ONE short paragraph in Spanish with the same information.

FORMAT YOUR ANSWER EXACTLY LIKE THIS:

Damage Summary (English):
<one short paragraph>

Damage Summary (Spanish):
<one short paragraph>

RULES:
- If the damage type is "dent", the Spanish term MUST be "abolladura".
- If the damage type is "scratch", Spanish term MUST be "arañazo" or "rascadura".
- If the damage type is "broken headlight" or "broken light", Spanish term MUST be "faro roto".
- Never use terms like "grieta" for dents.
- Keep tone concise, accurate, and based ONLY on the metadata and image.
- Do NOT add extra details that are not detected.
    """
    return prompt.strip()


def generate_description_qwen(metadata, image_paths):
    tokenizer, model = get_qwen()

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": build_prompt(metadata)}]
        }
    ]

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        conv = [{
            "role": "user",
            "content": [{"type": "image", "image": img}]
        }]

        vision_tokens = process_vision_info(conv)
        vision_tokens = [v for v in vision_tokens if v is not None]
        messages[0]["content"].extend(vision_tokens)

    text_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=350,
        temperature=0.2,
        repetition_penalty=1.05
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "assistant\n" in output_text:
        output_text = output_text.split("assistant\n", 1)[1].strip()
    elif "<assistant>" in output_text:
        output_text = output_text.split("<assistant>", 1)[1].strip()

    return output_text