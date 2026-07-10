import sys
import argparse
import numpy as np
import rawpy
import tifffile
from pathlib import Path
from scipy.ndimage import laplace

# --- CONFIGURATION ---
TOP_N_PATCHES = 6         
MIN_BRIGHTNESS = 4000     

def analyze_and_harvest(cr3_path: Path, output_dir: Path, save_png: bool, patch_size: int):
    print(f"Auditing & Harvesting: {cr3_path.name} (Size: {patch_size}x{patch_size})...")
    
    gt_dir = output_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. READ RAW
    try:
        with rawpy.imread(str(cr3_path)) as raw:
            rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, use_camera_wb=False, output_bps=16)
    except Exception as e:
        print(f"  [ERROR] {e}"); return

    # 2. GLOBAL ANALYSIS
    gray = (0.2126*rgb[:,:,0] + 0.7152*rgb[:,:,1] + 0.0722*rgb[:,:,2]).astype(np.float32)
    global_sharpness = laplace(gray).var()
    mean_r, mean_g, mean_b = np.mean(rgb, axis=(0,1))
    gain_r, gain_b = mean_g/mean_r, mean_g/mean_b
    
    global_report = [f"GLOBAL ANALYSIS: {cr3_path.name}", "="*50,
                     f"Global Sharpness (Laplace Var): {global_sharpness:.2f}",
                     f"AWB Gains: R={gain_r:.3f}, B={gain_b:.3f}\n",
                     "HARVESTED SLICES:"]

    # 3. HARVESTING BEST SLICES
    candidates = []
    stride = patch_size // 2
    
    for y in range(0, rgb.shape[0]-patch_size+1, stride):
        for x in range(0, rgb.shape[1]-patch_size+1, stride):
            p_gray = gray[y:y+patch_size, x:x+patch_size]
            p_rgb = rgb[y:y+patch_size, x:x+patch_size]
            
            sharpness = laplace(p_gray).var()
            noise = np.percentile(p_gray, 5)
            brightness = np.mean(p_rgb)
            
            if brightness > MIN_BRIGHTNESS:
                candidates.append({'sharpness': sharpness, 'noise': noise, 'y': y, 'x': x, 'data': p_rgb})

    candidates.sort(key=lambda x: x['sharpness'], reverse=True)
    best_patches = candidates[:TOP_N_PATCHES]

    # Save patches and update global report
    for i, p in enumerate(best_patches):
        pid = f"{cr3_path.stem}_{patch_size}x{patch_size}_top{i+1}"
        global_report.append(f"Slice {i+1}: Coords(y={p['y']}, x={p['x']}) | Sharpness: {p['sharpness']:.2f}")
        
        gt = p['data'].astype(np.float32)
        gt[:,:,0] *= gain_r
        gt[:,:,2] *= gain_b
        gt = np.clip(gt, 0, 65535).astype(np.uint16)
        
        np.save(gt_dir / f"{pid}.npy", gt)
        if save_png: tifffile.imwrite(gt_dir / f"{pid}.png", gt)
        
        # Local Report
        with open(gt_dir / f"{pid}_report.txt", 'w') as f:
            f.write(f"PATCH: {pid}\nSharpness: {p['sharpness']:.2f}\nNoise: {p['noise']:.2f}\nAWB(Global): R={gain_r:.3f}, B={gain_b:.3f}")

    with open(output_dir / f"{cr3_path.stem}_global_report.txt", 'w') as f:
        f.write('\n'.join(global_report))

    print(f"  [SUCCESS] Harvested {len(best_patches)} patches.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-png", action="store_true")
    parser.add_argument("-s", "--size", type=int, choices=[512, 1024], default=512)
    args = parser.parse_args()
    
    target = Path(args.input)
    out_dir = Path("Harvested_Dataset")
    out_dir.mkdir(exist_ok=True)
    
    files = [target] if target.is_file() else list(target.rglob('*.[cC][rR]3'))
    for f in files:
        analyze_and_harvest(f, out_dir, args.png, args.size)

if __name__ == "__main__":
    main()