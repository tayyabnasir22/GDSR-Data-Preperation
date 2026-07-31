from Models.BenchmarkType import BenchmarkType
from Utilities.DirectoryHelper import DirectoryHelper
from Utilities.PathManager import PathManager
import numpy as np
from PIL import Image
from numpy.lib.format import open_memmap

class ProcessingTOFDSRReal:
    TRAIN_FILE = PathManager.GetBasePath() + 'TOFDSR/TOFDSR_Train.txt'
    TEST_FILE = PathManager.GetBasePath() + 'TOFDSR/TOFDSR_Test.txt'

    BASE = PathManager.GetBasePath() + 'TOFDSR' # No backslash

    # HR resolution (RGB / HR GT depth). The real LR depth is bicubic-upsampled
    # to this grid, matching the TOFDSR benchmark protocol (DORNet).
    HR_H, HR_W = 384, 512

    @staticmethod
    def _GetPairs(base_path: str, base: str):
        # Each line: rgb_path, gt_path, lr_path (real LR from the phone ToF sensor)
        pairs = []
        with open(base_path, "r") as f:
            for index, line in enumerate(f):
                line = line.strip()          # remove newline and extra spaces
                parts = line.split(",")      # split by comma

                pairs.append(
                    (
                        base + parts[1].lstrip('TOFDC_split'),   # HR GT depth
                        base + parts[0].lstrip('TOFDC_split'),   # RGB
                        base + parts[2].lstrip('TOFDC_split'),   # real LR depth
                    )
                )

        return pairs

    @staticmethod
    def _LoadPaths():
        return ProcessingTOFDSRReal._GetPairs(ProcessingTOFDSRReal.TRAIN_FILE, ProcessingTOFDSRReal.BASE), ProcessingTOFDSRReal._GetPairs(ProcessingTOFDSRReal.TEST_FILE, ProcessingTOFDSRReal.BASE)

    @staticmethod
    def _LoadAllImages(paths):
        rgb_images = []
        depth_hr_images = []
        depth_lr_images = []

        for depth_hr_path, rgb_path, depth_lr_path in paths:
            # Load RGB (H, W, 3)
            rgb = np.array(Image.open(rgb_path).convert("RGB"))

            # Load HR GT Depth (H, W) in meters
            depth_hr = np.array(Image.open(depth_hr_path)) / 1000.0
            h, w = depth_hr.shape

            # Load real LR Depth, bicubic-upsample to the GT grid, convert to meters
            depth_lr = np.array(Image.open(depth_lr_path).resize((w, h), Image.BICUBIC)) / 1000.0

            rgb_images.append(rgb)
            depth_hr_images.append(depth_hr)
            depth_lr_images.append(depth_lr)

        return np.stack(rgb_images), np.stack(depth_hr_images), np.stack(depth_lr_images)

    @staticmethod
    def _NormalizeDepth(depth_maps):
        # Normalize Depths and generate min max map
        num_samples = depth_maps.shape[0]
        norm_depths = np.zeros_like(depth_maps, dtype=np.float32)
        minmax_list = np.zeros((num_samples,2), dtype=np.float32)

        for i in range(num_samples):
            d = depth_maps[i].astype(np.float32)
            d_min = d.min()
            d_max = d.max()
            
            # store max is first element and min is second
            minmax_list[i,0] = d_max
            minmax_list[i,1] = d_min
            
            # normalize to [0,1]
            norm_depths[i] = (d - d_min) / (d_max - d_min)

        return norm_depths, minmax_list

    @staticmethod
    def _NormalizeDepthWithMinMax(depth_maps, minmax_list):
        # Normalize depths using an externally provided per-sample (max, min).
        # The HR ground truth is normalized with the LR depth statistics so that
        # predictions can be de-normalized with the values that are actually
        # available at inference time in the real-world setting.
        num_samples = depth_maps.shape[0]
        norm_depths = np.zeros_like(depth_maps, dtype=np.float32)

        for i in range(num_samples):
            d = depth_maps[i].astype(np.float32)
            d_max = minmax_list[i, 0]
            d_min = minmax_list[i, 1]

            norm_depths[i] = (d - d_min) / (d_max - d_min)

        # HR GT can slightly exceed the LR min/max range -> keep values in [0,1]
        return np.clip(norm_depths, 0.0, 1.0)

    @staticmethod
    def _NormalizeRGB(images):
        # Normalize RGB
        images = images.astype(np.float32) / 255.0
        return images
    
    @staticmethod
    def _StandardizeRGB(images):
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

        images = (images - mean) / std

        return images

    @staticmethod
    def _GenerateDepthMaskBatch(depth_maps, min_depth=0.1, max_depth=6.0):
        mask = (depth_maps >= min_depth) & (depth_maps <= max_depth)
        return mask

    @staticmethod
    def ProcessBatches(pairs: list, prefix: str, save_path: str, batch_size: int):
        # 1. Create memmaps (disk-backed arrays)
        N = len(pairs)
        H, W = ProcessingTOFDSRReal.HR_H, ProcessingTOFDSRReal.HR_W

        imagesT_mm = open_memmap(save_path + prefix + '_images_split.npy', mode='w+', dtype=np.uint8, shape=(N, 3, H, W))
        imagesN_mm = open_memmap(save_path + prefix + '_images_norm_split.npy', mode='w+', dtype=np.float32, shape=(N, 3, H, W))
        imagesS_mm = open_memmap(save_path + prefix + '_images_stand_split.npy', mode='w+', dtype=np.float32, shape=(N, 3, H, W))

        depthT_mm  = open_memmap(save_path + prefix + '_depths_split.npy', mode='w+', dtype=np.float32, shape=(N, H, W))
        depthC_mm  = open_memmap(save_path + prefix + '_depths_clipped_split.npy', mode='w+', dtype=np.float32, shape=(N, H, W))
        depthN_mm  = open_memmap(save_path + prefix + '_depths_norm_split.npy', mode='w+', dtype=np.float32, shape=(N, H, W))

        depthLR_T_mm = open_memmap(save_path + prefix + '_depths_lr_split.npy', mode='w+', dtype=np.float32, shape=(N, H, W))
        depthLR_C_mm = open_memmap(save_path + prefix + '_depths_lr_clipped_split.npy', mode='w+', dtype=np.float32, shape=(N, H, W))
        depthLR_N_mm = open_memmap(save_path + prefix + '_depths_lr_norm_split.npy', mode='w+', dtype=np.float32, shape=(N, H, W))

        mask_mm    = open_memmap(save_path + prefix + '_mask_split.npy', mode='w+', dtype=bool, shape=(N, H, W))
        maskLR_mm  = open_memmap(save_path + prefix + '_mask_lr_split.npy', mode='w+', dtype=bool, shape=(N, H, W))
        minmax_mm  = open_memmap(save_path + prefix + '_minmax_split.npy', mode='w+', dtype=np.float32, shape=(N, 2))

        # 2. Process the batches
        print("Processing batches...")
        for start in range(0, N, batch_size):
            # 2.1. pick the batch and load data
            end = min(start + batch_size, N)
            paths = pairs[start:end]
            imagesT, depth_mapsT, depth_mapsLR_T = ProcessingTOFDSRReal._LoadAllImages(paths)

            # 2.2. Store the base image and depths
            imagesT = np.transpose(imagesT, (0, -1, 1 ,2))
            imagesT_mm[start:end] = imagesT
            depthT_mm[start:end]  = depth_mapsT
            depthLR_T_mm[start:end] = depth_mapsLR_T

            # 2.3. Normalize RGB using imagenet weights
            imagesN = ProcessingTOFDSRReal._NormalizeRGB(imagesT)
            imagesS = ProcessingTOFDSRReal._StandardizeRGB(imagesN)
            imagesN_mm[start:end] = imagesN
            imagesS_mm[start:end] = imagesS

            # 2.4. Generate masks for depth pixels out of range (HR GT and real LR)
            masks = ProcessingTOFDSRReal._GenerateDepthMaskBatch(depth_mapsT)
            masksLR = ProcessingTOFDSRReal._GenerateDepthMaskBatch(depth_mapsLR_T)
            mask_mm[start:end] = masks
            maskLR_mm[start:end] = masksLR

            # 2.5. Clip the depths between 0.1 and 6 m
            depth_mapsC = np.clip(depth_mapsT, 0.1, 6.0)
            depth_mapsLR_C = np.clip(depth_mapsLR_T, 0.1, 6.0)
            depthC_mm[start:end] = depth_mapsC
            depthLR_C_mm[start:end] = depth_mapsLR_C

            # 2.6. Normalize the LR depth with its own min/max (what is available
            #      at inference time) and store that min/max
            depth_mapsLR_N, minmax_list = ProcessingTOFDSRReal._NormalizeDepth(depth_mapsLR_C)
            depthLR_N_mm[start:end] = depth_mapsLR_N
            minmax_mm[start:end] = minmax_list

            # 2.7. Normalize the HR GT with the LR min/max so training targets
            #      and de-normalization are consistent with the LR input
            depth_mapsN = ProcessingTOFDSRReal._NormalizeDepthWithMinMax(depth_mapsC, minmax_list)
            depthN_mm[start:end] = depth_mapsN

    @staticmethod
    def GenerateNPYFiles(batch_size: int = 32):
        # 1. Load data paths
        ProcessingTOFDSRReal.TRAIN_FILE = PathManager.GetBasePath() + 'TOFDSR/TOFDSR_Train.txt'
        ProcessingTOFDSRReal.TEST_FILE = PathManager.GetBasePath() + 'TOFDSR/TOFDSR_Test.txt'
        ProcessingTOFDSRReal.BASE = PathManager.GetBasePath() + 'TOFDSR' # No backslash
        train_pairs, test_pairs = ProcessingTOFDSRReal._LoadPaths()

        # 2. Create the output path
        path = PathManager.GetBasePath() + BenchmarkType.TOFDSRDReal.name + '/'
        DirectoryHelper.ResetFolder(path)

        # 3. Process the data
        ProcessingTOFDSRReal.ProcessBatches(train_pairs, 'train', path, batch_size)
        ProcessingTOFDSRReal.ProcessBatches(test_pairs, 'test', path, batch_size)