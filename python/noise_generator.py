import numpy as np
from PIL import Image

def generate_sensor_noise(image_16bit: np.ndarray, shot_noise_coeff: float, read_noise_variance: float) -> np.ndarray:
    """
    Applies physical Poisson-Gaussian heteroscedastic noise to a 16-bit linear image.
    
    :param image_16bit: Clean ground truth 16-bit linear array.
    :param shot_noise_coeff: Beta1 (Controls signal-dependent photon noise).
    :param read_noise_variance: Beta2 (Controls constant electronic read noise).
    :return: Noisy 16-bit image array.
    """
    print("--- NOISE GENERATION PIPELINE ---")
    print(f"Applying Heteroscedastic Noise...")
    print(f"Shot Noise Coeff (B1) : {shot_noise_coeff}")
    print(f"Read Noise Var (B2)   : {read_noise_variance}")

    # 1. Normalize image to [0.0, 1.0] for stable floating-point math
    img_float = image_16bit.astype(np.float32) / 65535.0

    # 2. Calculate spatial variance map
    # Each pixel gets its own variance based on its intensity (Poisson nature)
    variance_map = (shot_noise_coeff * img_float) + read_noise_variance
    
    # Math safety: variance cannot be negative
    variance_map = np.clip(variance_map, 0.0, None)

    # 3. Calculate standard deviation (Sigma) for each pixel
    sigma_map = np.sqrt(variance_map)

    # 4. Generate the actual noise mask using normal distribution but with dynamic scale
    # np.random.normal can take an array for 'scale', generating unique noise per pixel
    noise_mask = np.random.normal(loc=0.0, scale=sigma_map, size=img_float.shape)

    # 5. Add noise to the clean image
    noisy_img_float = img_float + noise_mask

    # 6. Clip values back to valid [0.0, 1.0] range
    noisy_img_float = np.clip(noisy_img_float, 0.0, 1.0)

    # 7. Scale back to 16-bit space
    noisy_16bit = (noisy_img_float * 65535.0).astype(np.uint16)
    
    print("Noise applied successfully. Array ready for saving.\n")
    return noisy_16bit

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Create a synthetic 16-bit clean gradient patch (512x512) for testing
    print("Generating synthetic 512x512 linear gradient for testing...")
    clean_patch = np.linspace(0, 65535, 512*512, dtype=np.uint16).reshape(512, 512)
    
    # Emulate ISO 3200 noise levels
    BETA_1 = 0.005  # Shot noise (depends on light)
    BETA_2 = 0.0001 # Read noise (constant in shadows)
    
    noisy_patch = generate_sensor_noise(clean_patch, BETA_1, BETA_2)
    
    # Save to disk to visually inspect the difference
    Image.fromarray(clean_patch).save('synthetic_clean_gt.tiff')
    Image.fromarray(noisy_patch).save('synthetic_noisy_input.tiff')
    print("Saved 'synthetic_clean_gt.tiff' and 'synthetic_noisy_input.tiff' to disk.")