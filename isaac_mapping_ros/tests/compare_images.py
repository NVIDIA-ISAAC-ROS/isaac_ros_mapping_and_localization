import argparse
import sys
from PIL import Image
import numpy as np


def compare_images(image1_path, image2_path, pixel_threshold, diff_threshold):
    # Open the images
    try:
        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)
    except IOError:
        print("Error: Unable to open one or both images.", image1_path, image2_path)
        sys.exit(1)

    # Convert images to numpy arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    # Check if images have the same dimensions
    if arr1.shape != arr2.shape:
        print("Error: Images have different dimensions")
        sys.exit(1)

    # Calculate the absolute difference between the images
    diff = np.abs(arr1.astype(np.int16) - arr2.astype(np.int16))

    # Count pixels that differ more than the pixel_threshold
    diff_pixels = np.sum(diff > pixel_threshold)

    # Check if the number of different pixels exceeds the diff_threshold
    if diff_pixels > diff_threshold:
        print(f"Failure: {diff_pixels} pixels differ more than the threshold")
        sys.exit(1)
    else:
        print(f"Success: {diff_pixels} pixels differ more than the threshold")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Compare two PNG images and exit with" +
                                     " failure if the difference exceeds a threshold.", )
    parser.add_argument("--image1", help="Path to the first PNG image")
    parser.add_argument("--image2", help="Path to the second PNG image")
    parser.add_argument(
        "--pixel_threshold",
        type=int,
        help="Threshold for pixel difference (0-255)",
    )
    parser.add_argument(
        "--diff_threshold",
        type=int,
        help="Threshold for total number of differing pixels",
    )

    args = parser.parse_args()

    # Validate pixel_threshold
    if args.pixel_threshold < 0 or args.pixel_threshold > 255:
        print("Error: pixel_threshold must be between 0 and 255")
        sys.exit(1)

    # Validate diff_threshold
    if args.diff_threshold < 0:
        print("Error: diff_threshold must be a non-negative integer")
        sys.exit(1)

    compare_images(args.image1, args.image2, args.pixel_threshold, args.diff_threshold)


if __name__ == "__main__":
    main()
