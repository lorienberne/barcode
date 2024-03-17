import albumentations as A
import argparse
import os
import cv2
import random

def main(args):

    # Set seed for reproducibility
    random.seed(args.seed)

    input_directory = args.input_directory
    output_directory = args.output_directory

    # Delete the output directory if it exists
    if os.path.exists(output_directory):
        for root, dirs, files in os.walk(output_directory):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(output_directory)

    os.makedirs(output_directory, exist_ok=True)

    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".png"):

                # Define the augmentation pipeline
                transform = A.Compose(
                    [
                        A.RandomBrightnessContrast(p=args.random_brightness_contrast),
                        A.RandomGamma(p=args.random_gamma),
                        A.Blur(p=args.blur),
                        A.GaussNoise(p=args.gauss_noise),
                        A.RGBShift(p=args.rgb_shift),
                        A.RandomFog(p=args.random_fog),
                        A.RandomRain(p=args.random_rain),
                        A.RandomSnow(p=args.random_snow),
                        A.RandomSunFlare(p=args.random_sun_flare),
                        A.RandomShadow(p=args.random_shadow),
                        A.Rotate(limit=10, p=args.rotate),
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
    parser.add_argument(
        "-rn-br-cont",
        "--random-brightness-contrast",
        type=float,
        default=0.05,
        help="Random brightness contrast",
    )
    parser.add_argument(
        "-rn-gamma",
        "--random-gamma",
        type=float,
        default=0.05,
        help="Random gamma",
    )
    parser.add_argument(
        "-bl",
        "--blur",
        type=float,
        default=0.05,
        help="Blur",
    )
    parser.add_argument(
        "-gn",
        "--gauss-noise",
        type=float,
        default=0.05,
        help="Gauss noise",
    )
    parser.add_argument(
        "-rgb-sh",
        "--rgb-shift",
        type=float,
        default=0.05,
        help="RGB shift",
    )
    parser.add_argument(
        "-rn-fog",
        "--random-fog",
        type=float,
        default=0.05,
        help="Random fog",
    )
    parser.add_argument(
        "-rn-rain",
        "--random-rain",
        type=float,
        default=0.05,
        help="Random rain",
    )
    parser.add_argument(
        "-rn-snow",
        "--random-snow",
        type=float,
        default=0.05,
        help="Random snow",
    )
    parser.add_argument(
        "-rn-sun-flare",
        "--random-sun-flare",
        type=float,
        default=0.05,
        help="Random sun flare",
    )
    parser.add_argument(
        "-rn-shadow",
        "--random-shadow",
        type=float,
        default=0.05,
        help="Random shadow",
    )
    parser.add_argument(
        "-rot",
        "--rotate",
        type=float,
        default=0.05,
        help="Rotate",
    )
    parser.add_argument(
        "-seed",
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generator",
    )
    args = parser.parse_args()
    main(args)
