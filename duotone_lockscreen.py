import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import os


def get_dominant_colors(image, k=2):
    """Extracts the dominant colors from an image using KMeans clustering."""
    pixels = np.float32(image.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)
    _, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    dominant_colors = centers[sorted_indices]
    return dominant_colors[:2]  # Return the two most dominant colors


def duotone_gradient(color1, color2, width, height):
    """Creates a smooth gradient between two colors."""
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        ratio = y / float(height - 1)
        color = [int(color1[i] * (1 - ratio) + color2[i] * ratio) for i in range(3)]
        gradient[y, :] = color
    return gradient


def apply_noise(image, intensity=0.05):
    """Applies noise to an image."""
    noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


def process_gradient_image(
    gradient_image,
    noise_intensity,
    blur_radius,
    color_enhancement,
    contrast_enhancement,
):
    """Applies various post-processing effects to the gradient image."""
    gradient_array = np.array(gradient_image)
    noisy_gradient = apply_noise(gradient_array, intensity=noise_intensity)
    gradient_image = Image.fromarray(noisy_gradient)
    gradient_image = gradient_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    enhancer = ImageEnhance.Color(gradient_image)
    gradient_image = enhancer.enhance(color_enhancement)
    contrast_enhancer = ImageEnhance.Contrast(gradient_image)
    gradient_image = contrast_enhancer.enhance(contrast_enhancement)
    gradient_image = gradient_image.filter(ImageFilter.SMOOTH_MORE)
    return gradient_image


def main():
    # Define the dimensions for the new image
    width, height = 1080, 1920  # Standard smartphone resolution

    # Load the input image
    input_image_path = "input.jpg"  # Add the path to your input image
    input_image = Image.open(input_image_path)
    input_array = np.array(input_image)

    # Create a directory to save the output images
    output_dir = "./output_images"
    os.makedirs(output_dir, exist_ok=True)

    # Extract the two most dominant colors from the input image
    dominant_colors = get_dominant_colors(input_array)

    # Create the duotone gradient using the dominant colors
    gradient_array = duotone_gradient(
        dominant_colors[0], dominant_colors[1], width, height
    )

    # Convert to PIL Image for further processing
    gradient_image = Image.fromarray(gradient_array)

    # Define parameters for post-processing
    noise_intensity = 0.01
    blur_radius = 15
    color_enhancement = 1.2
    contrast_enhancement = 1.1

    # Apply post-processing effects
    final_gradient_image = process_gradient_image(
        gradient_image,
        noise_intensity,
        blur_radius,
        color_enhancement,
        contrast_enhancement,
    )

    # Save the final gradient image
    output_filename = f"{output_dir}/duotone_gradient.jpg"
    final_gradient_image.save(output_filename, quality=95)
    print(f"Saved: {output_filename}")


if __name__ == "__main__":
    main()
