import os
import argparse
import numpy as np
import rawpy
from scipy.ndimage import laplace
from PIL import Image
from pathlib import Path

# --- CONFIGURATION ---
VAR_THRESHOLD = 150.0      # Sharpness threshold (calibrated for normalized 8-bit scale)
PATCH_SIZE = 512           # Standard ML tile size
STRIDE = 256               # 50% overlap for dense extraction

# Noise model parameters (ISO 800 - 3200 equivalent)
BETA_1 = 0.005             # Shot noise coefficient (Poisson)
BETA_2 = 0.0001            # Read noise variance (Gaussian)

def generate_sensor_noise(image_16bit: np.ndarray) -> np.ndarray:
    """Applies physical Poisson-Gaussian heteroscedastic noise."""
    img_float = image_16bit.astype(np.float32) / 65535.0
    
    # Calculate spatial variance map based on pixel intensity
    variance_map = (BETA_1 * img_float) + BETA_2
    variance_map = np.clip(variance_map, 0.0, None)
    sigma_map = np.sqrt(variance_map)
    
    # Generate and apply noise
    noise_mask = np.random.normal(loc=0.0, scale=sigma_map, size=img_float.shape)
    noisy_img_float = np.clip(img_float + noise_mask, 0.0, 1.0)
    
    return (noisy_img_float * 65535.0).astype(np.uint16)

def process_file(cr3_path: Path, output_dir: Path, save_png: bool):
    print(f"\nProcessing: {cr3_path.name}")
    
    # Setup directories
    gt_dir = output_dir / "ground_truth"
    in_dir = output_dir / "input"
    gt_dir.mkdir(parents=True, exist_ok=True)
    in_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with rawpy.imread(str(cr3_path)) as raw:
            rgb_linear = raw.postprocess(
                gamma=(1, 1), 
                no_auto_bright=True, 
                use_camera_wb=False, 
                output_bps=16
            )
    except Exception as e:
        print(f"  [ERROR] Failed to read RAW: {e}")
        return

    # Convert to grayscale and normalize for variance math (8-bit scale)
    gray_array = (0.2126 * rgb_linear[:, :, 0] + 
                  0.7152 * rgb_linear[:, :, 1] + 
                  0.0722 * rgb_linear[:, :, 2]).astype(np.float32)
    gray_array = (gray_array / 65535.0) * 255.0

    h, w = gray_array.shape
    file_stem = cr3_path.stem
    extracted_count = 0

    # Sliding window extraction
    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            gray_patch = gray_array[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            variance = laplace(gray_patch).var()
            
            if variance > VAR_THRESHOLD:
                # Extract clean 16-bit RGB patch (Ground Truth)
                gt_patch = rgb_linear[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                
                # Generate degraded input patch (Noise added)
                in_patch = generate_sensor_noise(gt_patch)
                
                patch_id = f"{file_stem}_{y}_{x}"
                gt_path_base = gt_dir / patch_id
                in_path_base = in_dir / patch_id
                
                # 1. Save NPY (Golden standard for PyTorch/TensorFlow)
                np.save(f"{gt_path_base}_gt.npy", gt_patch)
                np.save(f"{in_path_base}_in.npy", in_patch)
                
                # 2. Save 16-bit PNG if requested (For visual inspection)
                if save_png:
                    Image.fromarray(gt_patch).save(f"{gt_path_base}_gt.png")
                    Image.fromarray(in_patch).save(f"{in_path_base}_in.png")
                    
                extracted_count += 1

    print(f"  [SUCCESS] Extracted {extracted_count} high-quality patches.")

def main():
    parser = argparse.ArgumentParser(description="RAW to Dataset Patch Extractor")
    parser.add_argument("input_path", type=str, help="Path to a .CR3 file or directory")
    parser.add_argument("--out", type=str, default="Dataset_v1", help="Output directory name")
    parser.add_argument("--png", action="store_true", help="Also save 16-bit PNG files for visual debugging")
    
    args = parser.parse_args()
    target_path = Path(args.input_path)
    output_dir = Path(args.out)

    if not target_path.exists():
        print(f"Error: Path {target_path} does not exist.")
        return

    if target_path.is_file() and target_path.suffix.lower() == '.cr3':
        process_file(target_path, output_dir, args.png)
    elif target_path.is_dir():
        cr3_files = list(target_path.rglob('*.[cC][rR]3'))
        print(f"Found {len(cr3_files)} .CR3 files.")
        for f in cr3_files:
            process_file(f, output_dir, args.png)
    else:
        print("Invalid input. Please provide a .CR3 file or a directory containing them.")

if __name__ == "__main__":
    main()