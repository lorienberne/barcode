import albumentations as A
import argparse
import os
import cv2


def main(args):

    input_directory = args.input_directory
    output_directory = args.output_directory

    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".png"):

                # Define the augmentation pipeline
                transform = A.Compose(
                    [
                        A.RandomBrightnessContrast(p=0.5),
                        A.RandomGamma(p=0.5),
                        A.Blur(p=0.5),
                        A.GaussNoise(p=0.5),
                        A.RGBShift(p=0.5),
                        A.RandomFog(p=0.5),
                        A.RandomRain(p=0.5),
                        A.RandomSnow(p=0.5),
                        A.RandomSunFlare(p=0.5),
                        A.RandomShadow(p=0.5),
                    ]
                )

                # Load the original image
                image = cv2.imread(os.path.join(root, file))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                for i in range(args.number_of_images):
                    # Apply the augmentation
                    augmented = transform(image=image)
                    image = augmented["image"]

                    # Save the degraded image
                    file_name = file.split(".")[0]
                    output_file = os.path.join(
                        output_directory, f"{file_name}_{i}.png"
                    )
                    cv2.imwrite(output_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EAN13 barcodes.")
    parser.add_argument(
        "-i",
        "--input-directory",
        type=str,
        default="./data/tmp/original",
        help="Directory containing the original images",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=str,
        default="./data/tmp/degraded",
        help="Directory to save the degraded images",
    )
    parser.add_argument(
        "-nr-images",
        "--number-of-images",
        type=int,
        default=50,
        help="Number of barcode images to generate",
    )
    args = parser.parse_args()
    main(args)
