import sys
import numpy as np
import rawpy
from scipy.ndimage import laplace
from pathlib import Path

# --- CONFIGURATION ---
VAR_THRESHOLD = 150.0      # Variance threshold for a patch to be considered "sharp"
CLIPPING_THRESHOLD = 0.05  # Maximum allowed percentage of clipped pixels (5%)
NOISE_WARNING_LIMIT = 50.0 # Threshold for noise estimation in flat areas

def analyze_patch_grid(gray_array, patch_size):
    """Slices the grid and returns statistics for a given patch size."""
    h, w = gray_array.shape
    stride = patch_size // 2 # 50% overlap
    
    variances = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = gray_array[y:y+patch_size, x:x+patch_size]
            var = laplace(patch).var()
            variances.append(var)
            
    variances = np.array(variances)
    good_patches = np.sum(variances > VAR_THRESHOLD)
    total_patches = len(variances)
    yield_pct = (good_patches / total_patches * 100) if total_patches > 0 else 0
    
    return total_patches, good_patches, yield_pct, variances

def process_cr3_file(file_path: Path):
    print(f"Analyzing: {file_path.name} ... ", end='', flush=True)
    
    # Create report path exactly next to the source file
    report_path = file_path.with_suffix('.txt')
    
    try:
        with rawpy.imread(str(file_path)) as raw:
            # Extract linear 16-bit data without AWB and gamma (values 0 - 65535)
            rgb_linear = raw.postprocess(
                gamma=(1, 1), 
                no_auto_bright=True, 
                use_camera_wb=False, 
                output_bps=16
            )
    except Exception as e:
        print(f"READ ERROR ({e})")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"FILE REJECTED: Critical RAW reading error.\nDetails: {e}\n")
        return

    # 1. Clipping analysis (16-bit space, max 65535)
    # Count pixels brighter than 64000 (approx. 97% brightness)
    clipping_ratio = np.sum(rgb_linear > 64000) / rgb_linear.size * 100
    
    # 2. AWB Evaluation (Multipliers based on Gray World assumption)
    mean_r = np.mean(rgb_linear[:, :, 0])
    mean_g = np.mean(rgb_linear[:, :, 1])
    mean_b = np.mean(rgb_linear[:, :, 2])
    
    wb_k_r = mean_g / mean_r if mean_r > 0 else 1.0
    wb_k_b = mean_g / mean_b if mean_b > 0 else 1.0
    
    # AWB Recommendation
    if 0.9 < wb_k_r < 1.1 and 0.9 < wb_k_b < 1.1:
        awb_rec = "Balance is close to neutral. No correction required."
    else:
        awb_rec = f"Recommended Multipliers (Gain): R={wb_k_r:.3f}, G=1.000, B={wb_k_b:.3f}"

    # 3. Convert to Grayscale (luminance) for spatial analysis
    # Canonical luminance coefficients applied to the 16-bit data
    gray_array = (0.2126 * rgb_linear[:, :, 0] + 
                  0.7152 * rgb_linear[:, :, 1] + 
                  0.0722 * rgb_linear[:, :, 2]).astype(np.float32)

    # --- CRITICAL SCALE FIX ---
    # Normalize 16-bit scale (0-65535) down to 8-bit scale (0-255)
    # This aligns the mathematical variance with our standard threshold limits
    gray_array = (gray_array / 65535.0) * 255.0
    # --------------------------

    # 4. Analyze 512x512 patches
    t_512, g_512, y_512, var_512 = analyze_patch_grid(gray_array, 512)
    
    # 5. Analyze 1024x1024 patches
    t_1024, g_1024, y_1024, var_1024 = analyze_patch_grid(gray_array, 1024)

    # 6. Noise Floor Estimation
    # Take the 5th percentile of variance (the flattest, non-textured areas)
    # Their variance represents the pure physical sensor noise
    noise_floor = np.percentile(var_512, 5) if len(var_512) > 0 else 0

    # --- REJECTION LOGIC ---
    is_rejected = False
    rejection_reasons = []

    if clipping_ratio > CLIPPING_THRESHOLD:
        is_rejected = True
        rejection_reasons.append(f"Critical overexposure: {clipping_ratio:.2f}% of the frame is clipped (norm < {CLIPPING_THRESHOLD}%).")
    
    if y_512 < 15.0 and y_1024 < 15.0:
        is_rejected = True
        rejection_reasons.append("Yield is too low. Less than 15% useful area (frame is out of focus or lacks texture).")
        
    if noise_floor > NOISE_WARNING_LIMIT:
        is_rejected = True
        rejection_reasons.append(f"Abnormally high shadow noise ({noise_floor:.1f}). Unsuitable for Ground Truth.")

    # --- REPORT GENERATION ---
    report = []
    report.append(f"AUTOMATED RAW FILE ANALYSIS: {file_path.name}")
    report.append("=" * 50)
    
    if is_rejected:
        report.append("STATUS: [ REJECTED ] ❌")
        report.append("\nREJECTION REASONS:")
        for r in rejection_reasons:
            report.append(f" - {r}")
    else:
        report.append("STATUS: [ APPROVED FOR DATASET ] ✅")
        
    report.append("\n--- 1. EXPOSURE AND NOISE ---")
    report.append(f"Clipping area              : {clipping_ratio:.3f}%")
    report.append(f"Estimated noise floor (ISO): {noise_floor:.1f} (Lower is cleaner)")
    
    report.append("\n--- 2. COLORIMETRY (LINEAR AWB) ---")
    report.append(f"Average channel levels     : R={mean_r:.0f} | G={mean_g:.0f} | B={mean_b:.0f}")
    report.append(f"Color shift recommendation : {awb_rec}")
    
    report.append("\n--- 3. SPATIAL CROPPING (SHARPNESS) ---")
    report.append("GRID 512x512 (Stride 256):")
    report.append(f"  Total patches   : {t_512}")
    report.append(f"  Useful patches  : {g_512} (Var > {VAR_THRESHOLD})")
    report.append(f"  Grid efficiency : {y_512:.1f}%")
    report.append(f"  Peak sharpness  : {np.max(var_512):.0f}" if len(var_512) > 0 else "  Peak sharpness  : 0")
    
    report.append("\nGRID 1024x1024 (Stride 512):")
    report.append(f"  Total patches   : {t_1024}")
    report.append(f"  Useful patches  : {g_1024}")
    report.append(f"  Grid efficiency : {y_1024:.1f}%")
    
    # Write report to disk
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
        
    if is_rejected:
        print("REJECTED ❌")
    else:
        print("APPROVED ✅")

def main():
    if len(sys.argv) < 2:
        print("Usage: python cr3_analyzer.py <path_to_file_or_directory>")
        sys.exit(1)
        
    target_path = Path(sys.argv[1])
    
    if not target_path.exists():
        print(f"Error: Path {target_path} does not exist.")
        sys.exit(1)
        
    if target_path.is_file():
        if target_path.suffix.lower() == '.cr3':
            process_cr3_file(target_path)
        else:
            print("The specified file is not a .CR3 file.")
            
    elif target_path.is_dir():
        print(f"Scanning directory: {target_path}")
        cr3_files = list(target_path.rglob('*.[cC][rR]3'))
        
        if not cr3_files:
            print("No .CR3 files found in the directory.")
            return
            
        print(f"Found files for analysis: {len(cr3_files)}\n")
        for f in cr3_files:
            process_cr3_file(f)

if __name__ == "__main__":
    main()