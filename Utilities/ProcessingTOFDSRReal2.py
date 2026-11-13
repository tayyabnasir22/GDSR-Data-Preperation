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
    PATH_PREFIX = 'TOFDC_split'

    # HR grid (RGB / HR GT depth) and the native grid of the filled phone-ToF LR depth
    HR_H, HR_W = 384, 512
    LR_H, LR_W = 144, 192

    # TOFDSR protocol (DORNet train_tofdsr.py / utils.tofdsr_calc_rmse, DuCos main_TOFDSR.py):
    #   - a GT pixel is valid only if 100 mm <= GT <= 5000 mm; loss and RMSE use valid pixels only
    #   - LR and GT are both normalized with the same FIXED range [0, 5000] mm
    VALID_MIN, VALID_MAX = 0.1, 5.0
    NORM_MIN, NORM_MAX = 0.0, 5.0

    @staticmethod
    def _StripPrefix(p: str):
        # str.lstrip('TOFDC_split') strips a *character set*, not a prefix. It only
        # works by accident because the next character is '/'. Strip the prefix explicitly.
        prefix = ProcessingTOFDSRReal.PATH_PREFIX
        return p[len(prefix):] if p.startswith(prefix) else '/' + p.lstrip('/')

    @staticmethod
    def _GetPairs(base_path: str, base: str):
        # Each line: rgb_path, gt_path, lr_path (LR_Filled = colorization-filled phone ToF depth)
        pairs = []
        with open(base_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]

                pairs.append(
                    (
                        base + ProcessingTOFDSRReal._StripPrefix(parts[1]),   # HR GT depth
                        base + ProcessingTOFDSRReal._StripPrefix(parts[0]),   # RGB
                        base + ProcessingTOFDSRReal._StripPrefix(parts[2]),   # real LR depth
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

            # Load HR GT depth (H, W), mm -> meters
            depth_hr = np.array(Image.open(depth_hr_path)).astype(np.float32) / 1000.0

            # Load real LR depth at its native resolution, mm -> meters
            depth_lr = np.array(Image.open(depth_lr_path)).astype(np.float32) / 1000.0

            expected_hr = (ProcessingTOFDSRReal.HR_H, ProcessingTOFDSRReal.HR_W)
            expected_lr = (ProcessingTOFDSRReal.LR_H, ProcessingTOFDSRReal.LR_W)
            if depth_hr.shape != expected_hr or rgb.shape[:2] != expected_hr:
                raise ValueError(f'Unexpected HR size {depth_hr.shape} / RGB {rgb.shape} in {depth_hr_path}')
            if depth_lr.shape != expected_lr:
                raise ValueError(f'Unexpected LR size {depth_lr.shape} in {depth_lr_path}')

            rgb_images.append(rgb)
            depth_hr_images.append(depth_hr)
            depth_lr_images.append(depth_lr)

        return np.stack(rgb_images), np.stack(depth_hr_images), np.stack(depth_lr_images)

    @staticmethod
    def _NormalizeFixed(depth_maps):
        # Fixed-range normalization. The constants are known at inference time, so the
        # prediction can always be mapped back to meters: d = d_norm * (max - min) + min.
        lo, hi = ProcessingTOFDSRReal.NORM_MIN, ProcessingTOFDSRReal.NORM_MAX
        return ((depth_maps - lo) / (hi - lo)).astype(np.float32)

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
    def _GenerateDepthMaskBatch(depth_maps):
        # IMPORTANT: must be computed on the RAW depth. After clipping to
        # [VALID_MIN, VALID_MAX] every pixel passes this test and the mask is useless.
        lo, hi = ProcessingTOFDSRReal.VALID_MIN, ProcessingTOFDSRReal.VALID_MAX
        mask = (depth_maps >= lo) & (depth_maps <= hi)
        return mask

    @staticmethod
    def ProcessBatches(pairs: list, prefix: str, save_path: str, batch_size: int):
        # 1. Create memmaps (disk-backed arrays)
        N = len(pairs)
        H, W = ProcessingTOFDSRReal.HR_H, ProcessingTOFDSRReal.HR_W
        LR_H, LR_W = ProcessingTOFDSRReal.LR_H, ProcessingTOFDSRReal.LR_W
        lo, hi = ProcessingTOFDSRReal.VALID_MIN, ProcessingTOFDSRReal.VALID_MAX

        imagesT_mm = open_memmap(save_path + prefix + '_images_split.npy', mode='w+', dtype=np.uint8, shape=(N, 3, H, W))
        imagesN_mm = open_memmap(save_path + prefix + '_images_norm_split.npy', mode='w+', dtype=np.float32, shape=(N, 3, H, W))
        imagesS_mm = open_memmap(save_path + prefix + '_images_stand_split.npy', mode='w+', dtype=np.float32, shape=(N, 3, H, W))

        depthT_mm  = open_memmap(save_path + prefix + '_depths_split.npy', mode='w+', dtype=np.float32, shape=(N, H, W))
        depthC_mm  = open_memmap(save_path + prefix + '_depths_clipped_split.npy', mode='w+', dtype=np.float32, shape=(N, H, W))
        depthN_mm  = open_memmap(save_path + prefix + '_depths_norm_split.npy', mode='w+', dtype=np.float32, shape=(N, H, W))

        depthLR_T_mm = open_memmap(save_path + prefix + '_depths_lr_split.npy', mode='w+', dtype=np.float32, shape=(N, LR_H, LR_W))
        depthLR_C_mm = open_memmap(save_path + prefix + '_depths_lr_clipped_split.npy', mode='w+', dtype=np.float32, shape=(N, LR_H, LR_W))
        depthLR_N_mm = open_memmap(save_path + prefix + '_depths_lr_norm_split.npy', mode='w+', dtype=np.float32, shape=(N, LR_H, LR_W))

        mask_mm    = open_memmap(save_path + prefix + '_mask_split.npy', mode='w+', dtype=bool, shape=(N, H, W))
        maskLR_mm  = open_memmap(save_path + prefix + '_mask_lr_split.npy', mode='w+', dtype=bool, shape=(N, LR_H, LR_W))
        minmax_mm  = open_memmap(save_path + prefix + '_minmax_split.npy', mode='w+', dtype=np.float32, shape=(N, 2))

        # 2. Process the batches
        valid_hr, valid_lr, empty_images = 0.0, 0.0, 0
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

            # 2.4. Validity masks from the RAW depths (HR GT and real LR)
            masks = ProcessingTOFDSRReal._GenerateDepthMaskBatch(depth_mapsT)
            masksLR = ProcessingTOFDSRReal._GenerateDepthMaskBatch(depth_mapsLR_T)
            mask_mm[start:end] = masks
            maskLR_mm[start:end] = masksLR
            valid_hr += masks.sum()
            valid_lr += masksLR.sum()
            empty_images += int((masks.reshape(masks.shape[0], -1).sum(axis=1) == 0).sum())

            # 2.5. Clip the depths to the valid range (0.1 - 5 m). Invalid GT pixels still
            #      hold a meaningless value here; they are excluded through the mask.
            depth_mapsC = np.clip(depth_mapsT, lo, hi)
            depth_mapsLR_C = np.clip(depth_mapsLR_T, lo, hi)
            depthC_mm[start:end] = depth_mapsC
            depthLR_C_mm[start:end] = depth_mapsLR_C

            # 2.6. Normalize LR and GT with the SAME fixed range [0, 5] m and store that
            #      range as the per-sample (max, min), so the data processor, the loss and
            #      the de-normalization in the metric keep working unchanged.
            depthLR_N_mm[start:end] = ProcessingTOFDSRReal._NormalizeFixed(depth_mapsLR_C)
            depthN_mm[start:end] = ProcessingTOFDSRReal._NormalizeFixed(depth_mapsC)
            minmax_mm[start:end, 0] = ProcessingTOFDSRReal.NORM_MAX   # max is the first element
            minmax_mm[start:end, 1] = ProcessingTOFDSRReal.NORM_MIN   # min is the second element

        # 3. Flush to disk and report how much of the GT is valid
        for m in (imagesT_mm, imagesN_mm, imagesS_mm, depthT_mm, depthC_mm, depthN_mm,
                  depthLR_T_mm, depthLR_C_mm, depthLR_N_mm, mask_mm, maskLR_mm, minmax_mm):
            m.flush()

        print(f'{prefix}: valid HR pixels {100 * valid_hr / (N * H * W):.1f}%, '
              f'valid LR pixels {100 * valid_lr / (N * LR_H * LR_W):.1f}%, '
              f'images without any valid GT pixel: {empty_images}')

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