import subprocess
import argparse

def main(number_of_images, save_path, file_path, mode, seed, degrade_args):
    original_save_path = save_path + "/original"
    degraded_save_path = save_path + "/degraded"
    # First, generate barcode images
    print("Generating barcode images...")
    subprocess.run([
        "python", "./utils/create_ean13.py",
        "--number-of-images", str(number_of_images),
        "--save-path", original_save_path,
        "--file-path", file_path,
        "--mode", mode,
        "--seed", str(seed)
    ], check=True)
    
    # Now, degrade the generated images
    print("Applying degradations to barcode images...")
    degrade_command = [
        "python", "./utils/degrade.py",
        "--input-directory", original_save_path,
        "--output-directory", degraded_save_path,
        "--seed", str(seed)
    ]
    # Append additional degrade arguments
    for arg, value in degrade_args.items():
        degrade_command.extend([arg, str(value)])
    
    subprocess.run(degrade_command, check=True)
    print("Process completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and degrade barcode images.")
    parser.add_argument("--number-of-images", type=int, default=10, help="Number of barcode images to generate and degrade")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save generated barcode images")
    parser.add_argument("--file-path", type=str, default="", help="Path to the text file with barcode numbers (only used in 'from_list' mode)")
    parser.add_argument("--mode", choices=["random", "from_list"], default="random", help="Mode to generate barcode numbers: 'random' or 'from_list'")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator")
    # Degradation arguments
    parser.add_argument("--random-brightness-contrast", type=float, default=0.2, help="Random brightness contrast")
    parser.add_argument("--number-of-degradations", type=int, default=30, help="Number of degradations of the original to generate")
    parser.add_argument("--random-gamma", type=float, default=0.5, help="Random gamma")
    parser.add_argument("--blur", type=float, default=0.5, help="Blur")
    parser.add_argument("--gauss-noise", type=float, default=0.1, help="Gauss noise")
    parser.add_argument("--rgb-shift", type=float, default=0.1, help="RGB shift")
    parser.add_argument("--random-fog", type=float, default=0.1, help="Random fog")
    parser.add_argument("--random-rain", type=float, default=0.1, help="Random rain")
    parser.add_argument("--random-snow", type=float, default=0.1, help="Random snow")
    parser.add_argument("--random-sun-flare", type=float, default=0.2, help="Random sun flare")
    parser.add_argument("--random-shadow", type=float, default=0.1, help="Random shadow")
    parser.add_argument("--rotate", type=float, default=5, help="Rotate")

    args = parser.parse_args()

    degrade_args = {
        "--random-brightness-contrast": args.random_brightness_contrast,
        "--number-of-degradations": args.number_of_degradations,
        "--random-gamma": args.random_gamma,
        "--blur": args.blur,
        "--gauss-noise": args.gauss_noise,
        "--rgb-shift": args.rgb_shift,
        "--random-fog": args.random_fog,
        "--random-rain": args.random_rain,
        "--random-snow": args.random_snow,
        "--random-sun-flare": args.random_sun_flare,
        "--random-shadow": args.random_shadow,
        "--rotate": args.rotate,
    }

    main(args.number_of_images, args.save_path, args.file_path, args.mode, args.seed, degrade_args)
