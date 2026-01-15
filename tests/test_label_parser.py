from pathlib import Path
from torchvision import datasets

from src.utils.car_label_parser import build_make_map_from_dataset, parse_predicted_class


def main():
    # Run this file from your repo root: ENMGT-5910-Project-Team-Project/
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "data" / "car_recognition" / "dataset"

    ds = datasets.ImageFolder(dataset_dir)
    make_map = build_make_map_from_dataset(ds.classes)

    # 1) Sanity check: first class from your dataset
    sample_label = ds.classes[0]
    print("\nSample label from dataset:")
    print(sample_label)
    print("Parsed:", parse_predicted_class(sample_label, make_map))

    # 2) Manual tests aligned with your dataset
    tests = [
        "AM General Hummer SUV 2000",
        "Aston Martin V8 Vantage Coupe 2012",
        "Land Rover Range Rover SUV 2012",
        "Honda Accord Sedan 2012",
        "Geo Metro Convertible 1993",
        "Mercedes-Benz E-Class Sedan 2012",
        "Rolls-Royce Ghost Sedan 2012",
        "smart fortwo Convertible 2012",
    ]

    print("\nManual tests:")
    for t in tests:
        print("\nInput:", t)
        print("Parsed:", parse_predicted_class(t, make_map))


if __name__ == "__main__":
    main()
