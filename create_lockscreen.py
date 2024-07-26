import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from scipy.interpolate import interp1d
import os


def create_smooth_gradient(colors, height):
    """Creates a smooth gradient between given colors."""
    gradient = np.zeros((height, 3), dtype=np.float32)
    n_colors = len(colors)
    t = np.linspace(0, 1, n_colors)

    for i in range(3):  # For each RGB channel
        channel_values = [color[i] for color in colors]
        f = interp1d(t, channel_values, kind="cubic")
        gradient[:, i] = f(np.linspace(0, 1, height))

    return gradient


def apply_noise(image, intensity=0.02):
    """Applies noise to an image."""
    noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


def extract_dominant_colors(image_array, k):
    """Extracts dominant colors using K-Means clustering."""
    pixels = np.float32(image_array.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Sort the dominant colors by brightness (V channel in HSV)
    sorted_centers = sorted(
        centers,
        key=lambda x: cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_BGR2HSV)[0, 0, 2],
        reverse=True,
    )

    return sorted_centers


def create_gradient_image(smooth_grad, width, height):
    """Creates a gradient image from the smooth gradient array."""
    gradient_image = Image.new("RGB", (width, height))

    for y in range(height):
        color = tuple(map(int, smooth_grad[y]))
        for x in range(width):
            gradient_image.putpixel((x, y), color)

    return gradient_image


def main():
    # Define the dimensions for the new image
    width, height = 1080, 1920  # Standard smartphone resolution

    # Load the input image
    input_image = Image.open("input.jpg")  # Add the path to your input image
    input_array = np.array(input_image)

    # Create a directory to save the output images
    output_dir = "./output_images"
    os.makedirs(output_dir, exist_ok=True)

    # Define the parameters to try
    k_values = [4]  # Number of clusters
    noise_intensities = [0.01, 0.02]  # Different noise intensities
    blur_radii = [20, 25]  # Different Gaussian blur radii
    color_enhancements = [1.2, 1.3]  # Different color enhancement factors
    contrast_enhancements = [1.0, 1.1]  # Different contrast enhancement factors

    # Iterate through all combinations of parameters
    for k in k_values:
        for noise_intensity in noise_intensities:
            for blur_radius in blur_radii:
                for color_enhancement in color_enhancements:
                    for contrast_enhancement in contrast_enhancements:
                        # Extract dominant colors
                        sorted_centers = extract_dominant_colors(input_array, k)

                        # Create a smooth gradient from the sorted dominant colors
                        smooth_grad = create_smooth_gradient(sorted_centers, height)

                        # Create a new gradient image
                        gradient_image = create_gradient_image(
                            smooth_grad, width, height
                        )

                        # Convert gradient image to numpy array for further processing
                        gradient_array = np.array(gradient_image)

                        # Apply subtle noise to the gradient image
                        noisy_gradient = apply_noise(
                            gradient_array, intensity=noise_intensity
                        )

                        # Convert back to PIL Image
                        gradient_image = Image.fromarray(noisy_gradient)

                        # Apply Gaussian blur for a smooth, iOS-like effect
                        gradient_image = gradient_image.filter(
                            ImageFilter.GaussianBlur(radius=blur_radius)
                        )

                        # Enhance the colors of the gradient image
                        enhancer = ImageEnhance.Color(gradient_image)
                        gradient_image = enhancer.enhance(color_enhancement)

                        # Enhance the contrast of the gradient image
                        contrast_enhancer = ImageEnhance.Contrast(gradient_image)
                        gradient_image = contrast_enhancer.enhance(contrast_enhancement)

                        # Apply additional smoothing to the gradient image
                        gradient_image = gradient_image.filter(ImageFilter.SMOOTH_MORE)

                        # Save the final gradient image
                        output_filename = f"{output_dir}/gradient_k{k}_noise{noise_intensity}_blur{blur_radius}_color{color_enhancement}_contrast{contrast_enhancement}.jpg"
                        gradient_image.save(output_filename, quality=95)

                        print(f"Saved: {output_filename}")


if __name__ == "__main__":
    main()
