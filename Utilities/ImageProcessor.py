import numpy as np
from PIL import Image

class ImageProcessor:
    @staticmethod
    def GenerateTestPatchesOverlap(images, depths, patch_size=336, stride=336):
        N, C, H, W = images.shape

        rgb_patches = []
        depth_patches = []

        for idx in range(N):

            img = images[idx]
            depth = depths[idx]

            # Compute start indices
            h_starts = list(range(0, H - patch_size + 1, stride))
            w_starts = list(range(0, W - patch_size + 1, stride))

            if h_starts[-1] != H - patch_size:
                h_starts.append(H - patch_size)

            if w_starts[-1] != W - patch_size:
                w_starts.append(W - patch_size)

            for i in h_starts:
                for j in w_starts:

                    img_patch = img[:, i:i+patch_size, j:j+patch_size]
                    depth_patch = depth[i:i+patch_size, j:j+patch_size]

                    rgb_patches.append(img_patch)
                    depth_patches.append(depth_patch)

        rgb_patches = np.stack(rgb_patches)
        depth_patches = np.stack(depth_patches)

        return rgb_patches, depth_patches
    
    @staticmethod
    def GenerateTestPatchesOverlap2(images, depths):
        N, C, H, W = images.shape
        rgb_patches = []
        depth_patches = []

        for idx in range(N):

            img = images[idx]
            depth = depths[idx]

            # Left patch
            rgb_patches.append(img[:, :, :320])
            depth_patches.append(depth[:, :320])

            # Right patch
            rgb_patches.append(img[:, :, 320:])
            depth_patches.append(depth[:, 320:])

        rgb_patches = np.stack(rgb_patches)
        depth_patches = np.stack(depth_patches)

        return rgb_patches, depth_patches
    
    @staticmethod
    def UpsampleBicubic(source, target_h, target_w):
        img = Image.fromarray(source)
        img = img.resize((target_w, target_h), Image.BICUBIC)
        return np.array(img)
    