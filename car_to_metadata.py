from pathlib import Path
from torchvision import datasets

from src.recognition.infer import get_car_info  # change if your folder name differs
from src.utils.car_label_parser import build_make_map_from_dataset, parse_predicted_class
from src.utils.metadata_builder import build_metadata

from src.description.generate import generate_description_qwen


def main():
    image_path = input("Enter image path for car recognition: ").strip()

    # 1) Run car recognition
    car_result = get_car_info(image_path)
    if car_result.get("status") != "success":
        print("\n❌ Car recognition failed:")
        print(car_result)
        return

    # 2) Build make_map from dataset classes
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "data" / "car_recognition" / "dataset"
    ds = datasets.ImageFolder(dataset_dir)
    make_map = build_make_map_from_dataset(ds.classes)

    # 3) Parse predicted class -> brand/model/year (Top-1 only)
    parsed = parse_predicted_class(car_result["predicted_class"], make_map)

    car_info = {
        "brand": parsed["brand"],
        "model": parsed["model"],
        "year": parsed["year"],
        "confidence": float(car_result["confidence"]),
    }

    # 4) Build metadata (damage/cost placeholders for now)
    metadata = build_metadata(car_info=car_info, damage_info=[], cost_info=None)

    print("\n✅ CAR INFO:")
    print(car_info)

    print("\n✅ METADATA:")
    print(metadata)

    # 5) Generate description (for now using the same single image)
    description = generate_description_qwen(metadata, [image_path])

    print("\n✅ DESCRIPTION OUTPUT:")
    print(description)


if __name__ == "__main__":
    main()