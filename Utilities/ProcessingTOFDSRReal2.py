from Models.BenchmarkType import BenchmarkType
from Utilities.DirectoryHelper import DirectoryHelper
from Utilities.PathManager import PathManager
import numpy as np
import os
from PIL import Image
from numpy.lib.format import open_memmap


class ProcessingTOFDSRReal:
    """TOFDSR real-world preprocessing (DORNet protocol).

    Layout under PathManager.GetBasePath() + 'TOFDSR/':
        TOFDSR_Train.txt, TOFDSR_Test.txt      (from DORNet's data/ folder)
        TOFDC_split/{Train,Test}/{RGB,GT,LR_Filled}/...
    Each list line is 'rgb,gt,lr' relative to that folder. Depth PNGs are
    uint16 millimetres. DORNet scores GT pixels in [100, 5000] mm only.
    """

    # Inferred from the first pair in GenerateNPYFiles (TOFDSR crops are not
    # all the same size, so stragglers are resized to the inferred grid).
    HR_H, HR_W = None, None
    LR_H, LR_W = None, None

    # DORNet valid range, metres
    MIN_DEPTH, MAX_DEPTH = 0.1, 5.0

    # ------------------------------------------------------------------
    @staticmethod
    def _GetPairs(list_file: str, base: str):
        # Each line: rgb_path, gt_path, lr_path (real LR, colorization-filled)
        pairs = []
        with open(list_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 3:
                    print('Bad line:', line)
                    continue
                rgb, gt, lr = [os.path.join(base, p) for p in parts]
                missing = [p for p in (rgb, gt, lr) if not os.path.isfile(p)]
                if missing:
                    print('Missing:', missing)
                    continue
                pairs.append((gt, rgb, lr))   # (HR GT depth, RGB, real LR depth)
        return pairs

    @staticmethod
    def _LoadPaths():
        base = PathManager.GetBasePath() + 'TOFDSR/'
        return (ProcessingTOFDSRReal._GetPairs(base + 'TOFDSR_Train.txt', base),
                ProcessingTOFDSRReal._GetPairs(base + 'TOFDSR_Test.txt', base))

    @staticmethod
    def _InferResolutions(first_pair):
        depth_hr_path, _, depth_lr_path = first_pair
        w, h = Image.open(depth_hr_path).size
        ProcessingTOFDSRReal.HR_H, ProcessingTOFDSRReal.HR_W = h, w
        w, h = Image.open(depth_lr_path).size
        ProcessingTOFDSRReal.LR_H, ProcessingTOFDSRReal.LR_W = h, w
        print(f'TOFDSR: HR {ProcessingTOFDSRReal.HR_H}x{ProcessingTOFDSRReal.HR_W}, '
              f'LR {ProcessingTOFDSRReal.LR_H}x{ProcessingTOFDSRReal.LR_W}')

    @staticmethod
    def _LoadAllImages(paths):
        H, W = ProcessingTOFDSRReal.HR_H, ProcessingTOFDSRReal.HR_W
        LR_H, LR_W = ProcessingTOFDSRReal.LR_H, ProcessingTOFDSRReal.LR_W

        rgb_images, depth_hr_images, depth_lr_images = [], [], []
        for depth_hr_path, rgb_path, depth_lr_path in paths:
            rgb = np.array(Image.open(rgb_path).convert("RGB"))
            depth_hr = np.array(Image.open(depth_hr_path)).astype(np.float32) / 1000.0   # mm -> m
            depth_lr = np.array(Image.open(depth_lr_path)).astype(np.float32) / 1000.0   # mm -> m

            # Keep every sample on the inferred grids
            if rgb.shape[:2] != (H, W):
                rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.BICUBIC))
            if depth_hr.shape != (H, W):
                depth_hr = np.array(Image.fromarray(depth_hr).resize((W, H), Image.NEAREST))
            if depth_lr.shape != (LR_H, LR_W):
                depth_lr = np.array(Image.fromarray(depth_lr).resize((LR_W, LR_H), Image.BICUBIC))

            rgb_images.append(rgb)
            depth_hr_images.append(depth_hr)
            depth_lr_images.append(depth_lr)
        return np.stack(rgb_images), np.stack(depth_hr_images), np.stack(depth_lr_images)

    # ------------------------------------------------------------------
    @staticmethod
    def _NormalizeDepth(depth_maps):
        num_samples = depth_maps.shape[0]
        norm_depths = np.zeros_like(depth_maps, dtype=np.float32)
        minmax_list = np.zeros((num_samples, 2), dtype=np.float32)
        for i in range(num_samples):
            d = depth_maps[i].astype(np.float32)
            d_min, d_max = d.min(), d.max()
            minmax_list[i, 0] = d_max   # max first, min second
            minmax_list[i, 1] = d_min
            if d_max - d_min == 0:
                print('Bug: constant LR depth at sample', i)
                d_max = d_min + 1e-6
            norm_depths[i] = (d - d_min) / (d_max - d_min)
        return norm_depths, minmax_list

    @staticmethod
    def _NormalizeDepthWithMinMax(depth_maps, minmax_list):
        num_samples = depth_maps.shape[0]
        norm_depths = np.zeros_like(depth_maps, dtype=np.float32)
        for i in range(num_samples):
            d = depth_maps[i].astype(np.float32)
            d_max, d_min = minmax_list[i, 0], minmax_list[i, 1]
            if d_max - d_min == 0:
                print('Bug: constant range at sample', i)
                d_max = d_min + 1e-6
            norm_depths[i] = (d - d_min) / (d_max - d_min)
        return np.clip(norm_depths, 0.0, 1.0)

    @staticmethod
    def _NormalizeRGB(images):
        return images.astype(np.float32) / 255.0

    @staticmethod
    def _StandardizeRGB(images):
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        return (images - mean) / std

    @staticmethod
    def _GenerateDepthMaskBatch(depth_maps):
        lo, hi = ProcessingTOFDSRReal.MIN_DEPTH, ProcessingTOFDSRReal.MAX_DEPTH
        return (depth_maps >= lo) & (depth_maps <= hi)

    # ------------------------------------------------------------------
    @staticmethod
    def ProcessBatches(pairs, prefix, save_path, batch_size):
        N = len(pairs)
        H, W = ProcessingTOFDSRReal.HR_H, ProcessingTOFDSRReal.HR_W
        LR_H, LR_W = ProcessingTOFDSRReal.LR_H, ProcessingTOFDSRReal.LR_W
        lo, hi = ProcessingTOFDSRReal.MIN_DEPTH, ProcessingTOFDSRReal.MAX_DEPTH

        imagesT_mm = open_memmap(save_path + prefix + '_images_split.npy', 'w+', np.uint8, (N, 3, H, W))
        imagesN_mm = open_memmap(save_path + prefix + '_images_norm_split.npy', 'w+', np.float32, (N, 3, H, W))
        imagesS_mm = open_memmap(save_path + prefix + '_images_stand_split.npy', 'w+', np.float32, (N, 3, H, W))

        depthT_mm = open_memmap(save_path + prefix + '_depths_split.npy', 'w+', np.float32, (N, H, W))
        depthC_mm = open_memmap(save_path + prefix + '_depths_clipped_split.npy', 'w+', np.float32, (N, H, W))
        depthN_mm = open_memmap(save_path + prefix + '_depths_norm_split.npy', 'w+', np.float32, (N, H, W))

        depthLR_T_mm = open_memmap(save_path + prefix + '_depths_lr_split.npy', 'w+', np.float32, (N, LR_H, LR_W))
        depthLR_C_mm = open_memmap(save_path + prefix + '_depths_lr_clipped_split.npy', 'w+', np.float32, (N, LR_H, LR_W))
        depthLR_N_mm = open_memmap(save_path + prefix + '_depths_lr_norm_split.npy', 'w+', np.float32, (N, LR_H, LR_W))

        mask_mm = open_memmap(save_path + prefix + '_mask_split.npy', 'w+', bool, (N, H, W))
        maskLR_mm = open_memmap(save_path + prefix + '_mask_lr_split.npy', 'w+', bool, (N, LR_H, LR_W))
        minmax_mm = open_memmap(save_path + prefix + '_minmax_split.npy', 'w+', np.float32, (N, 2))

        valid_hr, valid_lr = 0.0, 0.0
        print("Processing batches...")
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            imagesT, depth_mapsT, depth_mapsLR_T = ProcessingTOFDSRReal._LoadAllImages(pairs[start:end])

            imagesT = np.transpose(imagesT, (0, -1, 1, 2))
            imagesT_mm[start:end] = imagesT
            depthT_mm[start:end] = depth_mapsT
            depthLR_T_mm[start:end] = depth_mapsLR_T

            imagesN = ProcessingTOFDSRReal._NormalizeRGB(imagesT)
            imagesS = ProcessingTOFDSRReal._StandardizeRGB(imagesN)
            imagesN_mm[start:end] = imagesN
            imagesS_mm[start:end] = imagesS

            masks = ProcessingTOFDSRReal._GenerateDepthMaskBatch(depth_mapsT)
            masksLR = ProcessingTOFDSRReal._GenerateDepthMaskBatch(depth_mapsLR_T)
            mask_mm[start:end] = masks
            maskLR_mm[start:end] = masksLR
            valid_hr += masks.sum(); valid_lr += masksLR.sum()

            depth_mapsC = np.clip(depth_mapsT, lo, hi)
            depth_mapsLR_C = np.clip(depth_mapsLR_T, lo, hi)
            depthC_mm[start:end] = depth_mapsC
            depthLR_C_mm[start:end] = depth_mapsLR_C

            # LR normalized with its own min/max (available at inference);
            # HR GT normalized with the same statistics.
            depth_mapsLR_N, minmax_list = ProcessingTOFDSRReal._NormalizeDepth(depth_mapsLR_C)
            depthLR_N_mm[start:end] = depth_mapsLR_N
            minmax_mm[start:end] = minmax_list
            depthN_mm[start:end] = ProcessingTOFDSRReal._NormalizeDepthWithMinMax(depth_mapsC, minmax_list)

        for m in (imagesT_mm, imagesN_mm, imagesS_mm, depthT_mm, depthC_mm, depthN_mm,
                  depthLR_T_mm, depthLR_C_mm, depthLR_N_mm, mask_mm, maskLR_mm, minmax_mm):
            m.flush()

        print(f'{prefix}: valid HR pixels {100 * valid_hr / (N * H * W):.1f}%, '
              f'valid LR pixels {100 * valid_lr / (N * LR_H * LR_W):.1f}%')

    @staticmethod
    def GenerateNPYFiles(batch_size: int = 32):
        train_pairs, test_pairs = ProcessingTOFDSRReal._LoadPaths()
        print("Train samples:", len(train_pairs))
        print("Test samples:", len(test_pairs))

        ProcessingTOFDSRReal._InferResolutions(train_pairs[0])

        path = PathManager.GetBasePath() + BenchmarkType.TOFDSRReal.name + '/'
        DirectoryHelper.ResetFolder(path)

        ProcessingTOFDSRReal.ProcessBatches(train_pairs, 'train', path, batch_size)
        ProcessingTOFDSRReal.ProcessBatches(test_pairs, 'test', path, batch_size)