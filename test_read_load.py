from pyzbar.pyzbar import decode
from PIL import Image
from termcolor import colored, cprint
import pandas as pd
import json
import os
import time


def load_barcode(image_path: str) -> Image:
    img = Image.open(image_path)
    return img


def decode_barcode(image: Image) -> str:
    decoded = decode(image)
    return decoded[0].data.decode("utf-8")


if __name__ == "__main__":
    avg_times_dict = {}
    for _, _, barcode_path in os.walk("./data/different_barcodes"):
        barcode_paths = [
            path.split("_image_")[1].replace(".png", "") for path in barcode_path
        ]
        for path in barcode_paths:

            avg_times_dict[path] = {}

            times_load = []
            for i in range(10):
                start = time.time()
                img = load_barcode("./data/different_barcodes/barcode_image_ean.png")
                times_load.append(round(time.time() - start, 5))
            average_time_load = sum(times_load) / len(times_load)
            avg_times_dict[path]["load"] = average_time_load

            times_decode = []
            for i in range(10):
                start = time.time()
                data = decode_barcode(img)
                times_decode.append(round(time.time() - start, 4))
            average_time_decode = sum(times_decode) / len(times_decode)
            avg_times_dict[path]["decode"] = average_time_decode
            cprint(f"{path}", "green")


pretty_printed_json = json.dumps(avg_times_dict, indent=4)
# print(pretty_printed_json)

print(pd.DataFrame(avg_times_dict).T.to_markdown())
