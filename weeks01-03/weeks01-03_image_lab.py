# Name: Essar Singh
# Roll No: 2k24/CSME/16
# GitHub Username: 2K24CSE16
# Course: Weeks 1-3 Lab: Image Pixels, Filters, and Edges
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def inspect_image(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    height, width, channels = img.shape
    return {
        "width": width,
        "height": height,
        "channels": channels,
        "shape": [int(height), int(width), int(channels)],
        "pixel_count": width * height,
        "estimated_bytes": width * height * channels,
        "color_order": "BGR"
    }

def create_pixel_views(image_path: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    b, g, r = cv2.split(img)
    zeros = np.zeros_like(b)
    blue_img = cv2.merge([b, zeros, zeros])
    green_img = cv2.merge([zeros, g, zeros])
    red_img = cv2.merge([zeros, zeros, r])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    downsampled = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_AREA)
    down_h, down_w = downsampled.shape[:2]
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.flatten()
    titles = ["Original", "Red Channel", "Green Channel", "Blue Channel", "Grayscale", f"Half Size {down_w}x{down_h}"]
    imgs = [img, red_img, green_img, blue_img, gray_bgr, downsampled]
    for ax, title, im in zip(axs, titles, imgs):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    output_path = os.path.join(output_dir, "pixel_views.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    return {"original_size": [w, h], "downsampled_size": [down_w, down_h], "output_path": output_path}

def create_adjustments(image_path: str, output_dir: str, brightness_delta: int = 40, contrast_factor: float = 1.5, threshold: int = 127) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be in 0-255")
    gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    brighter = np.clip(gray.astype(int) + brightness_delta, 0, 255).astype(np.uint8)
    contrast = np.clip(gray.astype(float) * contrast_factor, 0, 255).astype(np.uint8)
    _, thresh_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    data = [(gray, "Grayscale"), (brighter, f"Brighter +{brightness_delta}"), (contrast, f"Contrast x{contrast_factor}"), (thresh_img, f"Threshold {threshold}")]
    for ax, (im, title) in zip(axs, data):
        ax.imshow(im, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title); ax.axis('off')
    plt.tight_layout()
    output_path = os.path.join(output_dir, "adjustments.png")
    plt.savefig(output_path, dpi=150); plt.close()
    return {"brightness_delta": brightness_delta, "contrast_factor": contrast_factor, "threshold": threshold, "output_path": output_path}

def create_blur_and_edges(image_path: str, output_dir: str, kernel_size: int = 5) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be positive odd")
    gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (kernel_size, kernel_size))
    sobel_orig = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3))
    sobel_blur = cv2.convertScaleAbs(cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=3))
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    data = [(gray, "Grayscale"), (blurred, f"Mean Blur k={kernel_size}"), (sobel_orig, "Sobel Original"), (sobel_blur, "Sobel on Blurred")]
    for ax, (im, title) in zip(axs, data):
        ax.imshow(im, cmap='gray'); ax.set_title(title); ax.axis('off')
    plt.tight_layout()
    output_path = os.path.join(output_dir, "blur_and_edges.png")
    plt.savefig(output_path, dpi=150); plt.close()
    return {"kernel_size": kernel_size, "output_path": output_path}

def run_lab(image_path: str, output_dir: str) -> dict:
    return {"task1": inspect_image(image_path), "task2": create_pixel_views(image_path, output_dir), "task3": create_adjustments(image_path, output_dir), "task4": create_blur_and_edges(image_path, output_dir)}

def main() -> None:
    os.makedirs("outputs", exist_ok=True)
    results = run_lab("images/original.jpg", "outputs")
    print("Lab completed", results)

if __name__ == "__main__":
    main()