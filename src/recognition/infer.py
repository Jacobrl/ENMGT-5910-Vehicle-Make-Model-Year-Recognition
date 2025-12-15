import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from pathlib import Path
from PIL import Image

# LOAD YOUR FINE-TUNED RESNET-50
def load_model(weights_path: str, num_classes: int):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# LOAD IMAGENET MODEL FOR CAR/NON-CAR CHECK
imagenet_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
imagenet_model.eval()

imagenet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

CAR_IMAGENET_CLASSES = {
    751,  # sports car
    817,  # convertible
    511,  # minivan
    717,  # race car
    468,  # police van
    436,  # cab
    609,  # pickup truck
}


def imagenet_is_car(image: Image.Image):
    x = imagenet_transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = imagenet_model(x)
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence, cls = torch.max(probs, 1)

    return cls.item() in CAR_IMAGENET_CLASSES, float(confidence.item())

# 3. PREPROCESS IMAGE (same as training)
def preprocess_image(image_path: str):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0), image

# 4. RUN TOP-5 PREDICTION
def predict(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top5_prob, top5_idx = probs.topk(5, dim=1)

    results = [
        {
            "label": class_names[idx],
            "probability": float(prob)
        }
        for idx, prob in zip(top5_idx[0], top5_prob[0])
    ]
    return results

# 5. MAIN PIPELINE FUNCTION (NO PARSER)
def get_car_info(image_path: str):
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "car_recognition" / "dataset"
    model_path = base_dir / "resnet50_car_recognition_best.pth"

    dataset = datasets.ImageFolder(data_dir)
    class_names = dataset.classes
    num_classes = len(class_names)

    # Load fine-tuned model
    model = load_model(str(model_path), num_classes)

    # Preprocess
    image_tensor, raw_img = preprocess_image(image_path)

    # Stage 1 → ImageNet Car Filter
    is_car, car_conf = imagenet_is_car(raw_img)

    if not is_car:
        return {
            "status": "error",
            "reason": "Image does not contain a car.",
            "imagenet_confidence": car_conf
        }

    # Stage 2 → Fine-tuned classifier
    predictions = predict(model, image_tensor, class_names)

    top1 = predictions[0]
    top1_label = top1["label"]
    top1_conf = top1["probability"] * 100

    # Threshold check
    if top1_conf < 70:
        return {
            "status": "error",
            "reason": "Car detected, but the image is unclear. Please upload a better image.",
            "predicted_class": top1_label,
            "confidence": top1_conf
        }

    # Success
    return {
        "status": "success",
        "predicted_class": top1_label,
        "confidence": top1_conf,
        "top5_predictions": [
            {
                "label": p["label"],
                "confidence": p["probability"] * 100
            }
            for p in predictions
        ]
    }


if __name__ == "__main__":
    image_path = input("Enter path to image: ").strip()
    result = get_car_info(image_path)

    # Case 1 — Not a car
    if result["status"] == "error" and result.get("reason") == "Image does not contain a car.":
        print("Not a car image!!!")
        print(f"ImageNet confidence: {result['imagenet_confidence']:.2f}%")
        exit()

    # Case 2 — Low-quality car (<70% confidence)
    if result["status"] == "error":
        print("Car detected but the image is unclear.")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print("Please upload a better image!!!")
        exit()

    # Case 3 — Success (Top-5 predictions, previous iteration style)
    print("Top Predictions:")
    for i, pred in enumerate(result["top5_predictions"], 1):
        print(f"{i}. {pred['label']} — {pred['confidence']:.2f}%")