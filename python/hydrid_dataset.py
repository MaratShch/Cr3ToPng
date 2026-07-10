import numpy as np
import imageio.v2 as imageio
from torch.utils.data import Dataset

class HybridNoiseDataset(Dataset):
    def __init__(self, file_paths):
        """
        file_paths: List of file paths. It can contain a mix of 
        your reference .npy (16-bit linear) and downloaded .png (8-bit sRGB) files.
        """
        self.file_paths = file_paths
        self.gamma = 2.2

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = str(self.file_paths[idx])
        
        if path.endswith('.npy'):
            # 1. RAW DOMAIN (16-bit)
            # Load the array and normalize.
            # The array is ALREADY linear, no gamma correction is needed.
            img = np.load(path).astype(np.float32)
            img_tensor = img / 65535.0
            
        elif path.endswith('.png'):
            # 2. INTERNET DOMAIN (8-bit)
            # Load the 8-bit PNG
            img = imageio.imread(path).astype(np.float32)
            
            # Normalize to [0.0, 1.0] range
            img_normalized = img / 255.0
            
            # DE-GAMMIFICATION (Return to physical linearity)
            # Now this tensor is mathematically compatible with the linear RAW data
            img_tensor = np.power(img_normalized, self.gamma)
            
        else:
            raise ValueError(f"Unsupported format encountered: {path}")

        # The img_tensor is now ready for noise generation and model training.
        # Format: [H, W, C], Range: [0.0, 1.0], Color Space: Linear
        return img_tensor