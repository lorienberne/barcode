import barcode
from barcode.writer import ImageWriter
from tqdm import tqdm
import os
import argparse
import random


def create_barcode(data: str, b_type: str, idx: int, folder_path:str) -> None:
    folder_path = folder_path
    os.makedirs(folder_path, exist_ok=True)
    file_name = f"{idx}_{data}_ean13"
    file_path = os.path.join(folder_path, file_name)
    try:
        barcode_obj = barcode.get_barcode_class(b_type)
        barcode_image = barcode_obj(data, writer=ImageWriter())
        barcode_image.save(file_path)
    except Exception as e:
        print(f"Failed to create barcode for {b_type} due to {e}")


def generate_code(digits: int) -> str:
    base = str(random.randint(10 ** (digits - 2), 10 ** (digits - 1) - 1))
    return barcode.EAN13(base).get_fullcode()


def main(num_images, file_path, mode, seed, folder_path:str):
    random.seed(seed)  # Set a default seed for reproducibility
    barcode_type = "ean13"
    if mode == "from_list":
        idx = 1
        with open(file_path, "r") as file:
            for line in tqdm(
                file, total=num_images, desc="Generating Barcodes from List"
            ):
                data = line.strip()
                create_barcode(data, barcode_type, idx, folder_path=folder_path)
                idx += 1
                if idx > num_images:
                    break
    elif mode == "random":
        digits = 13
        for idx in tqdm(range(1, num_images + 1), desc="Generating Random Barcodes"):
            data = generate_code(digits)
            create_barcode(data, barcode_type, idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EAN13 barcodes.")
    parser.add_argument(
        "-nr-images",
        "--number-of-images",
        type=int,
        default=50000,
        help="Number of barcode images to generate",
    )
    parser.add_argument(
        "-save-path",
        "--save-path",
        type=str,
        help="Path to folder where to save pictures",
        default="./data/tmp/original",
    )
    parser.add_argument(
        "-file-path",
        "--file-path",
        type=str,
        help="Path to the text file with barcode numbers",
        default="",
    )
    parser.add_argument(
        "-mode",
        "--mode",
        choices=["random", "from_list"],
        default="random",
        help="Mode to generate barcode numbers: 'random' or 'from_list'",
    )
    parser.add_argument(
        "-seed", "--seed", type=int, default=42, help="Seed for random number generator"
    )
    args = parser.parse_args()

    main(args.number_of_images, args.file_path, args.mode, args.seed, folder_path=args.save_path)
