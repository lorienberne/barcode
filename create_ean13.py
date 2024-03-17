import barcode
from barcode.writer import ImageWriter
from tqdm import tqdm
import random
import os
import argparse


def create_barcode(data: str, b_type: str, idx: int) -> None:
    folder_path = "./data/ean13"
    os.makedirs(folder_path, exist_ok=True)
    file_name = f"{idx}_{data}_ean13"
    file_path = os.path.join(
        folder_path, file_name + ".png"
    )  # Ensure the file name has the .png extension
    try:
        barcode_obj = barcode.get_barcode_class(b_type)
        barcode_image = barcode_obj(data, writer=ImageWriter())
        barcode_image.save(file_path)
    except Exception as e:
        print(f"Failed to create barcode for {b_type} due to {e}")


def generate_code(digits: int) -> str:
    base = str(random.randint(10 ** (digits - 2), 10 ** (digits - 1) - 1))
    return barcode.EAN13(base).get_fullcode()


def main(num_images):
    barcode_type = "ean13"
    digits = 13
    dataset = []
    for idx in tqdm(range(1, num_images + 1), desc="Generating Barcodes"):
        data = generate_code(digits)
        dataset.append(data)
        create_barcode(data, barcode_type, idx)
    with open("./data/ean13/dataset.txt", "w") as file:
        for data in dataset:
            file.write(f"{data}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EAN13 barcodes.")
    parser.add_argument(
        "-nr-images",
        "--number-of-images",
        type=int,
        default=50000,
        help="Number of barcode images to generate",
    )
    args = parser.parse_args()
    main(args.number_of_images)
