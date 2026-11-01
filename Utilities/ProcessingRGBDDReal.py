from Models.BenchmarkType import BenchmarkType
from Utilities.DirectoryHelper import DirectoryHelper
from Utilities.PathManager import PathManager
import numpy as np
import os
from PIL import Image
from numpy.lib.format import open_memmap

class ProcessingRGBDDReal:
    # HR resolution (RGB / HR GT depth) and LR resolution (phone ToF depth)
    HR_H, HR_W = 384, 512
    LR_H, LR_W = 144, 192

    @staticmethod
    def _LoadPairPaths(path: str):
        pairs = {}
        for index, obj in enumerate(os.walk(path)):
            root, _, files = obj
            depth_hr = None
            depth_lr = None
            rgb = None
            for file in files:
                if file.endswith('HR_gt.png'):
                    depth_hr = os.path.join(root, file)
                if file.endswith('LR_fill_depth.png'):
                    depth_lr = os.path.join(root, file)
                if file.endswith('RGB.jpg'):
                    rgb = os.path.join(root, file)

            if rgb is None or depth_hr is None or depth_lr is None:
                print(root)
            else:
                pairs[index] = (depth_hr, rgb, depth_lr)

        return pairs

    @staticmethod
    def _GenerateDepthMaskBatch(depth_maps, min_depths: list[float], max_depths: list[float]):
        min_depths = np.asarray(min_depths, dtype=np.float32).reshape(-1, 1, 1)
        max_depths = np.asarray(max_depths, dtype=np.float32).reshape(-1, 1, 1)
        mask = (depth_maps >= min_depths) & (depth_maps <= max_depths)
        return mask, np.clip(depth_maps, min_depths, max_depths)
    
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
            
            if d_max - d_min == 0:
                print('Bug')


            # normalize to [0,1]
            norm_depths[i] = (d - d_min) / (d_max - d_min)

        return norm_depths, minmax_list

    @staticmethod
    def _NormalizeDepthWithMinMax(depth_maps, minmax_list):
        # Normalize depths using an externally provided per-sample (max, min).
        # For the real-world setting the HR ground truth is normalized with the
        # LR depth statistics so that predictions can be de-normalized with the
        # same values that were available at inference time.
        num_samples = depth_maps.shape[0]
        norm_depths = np.zeros_like(depth_maps, dtype=np.float32)

        for i in range(num_samples):
            d = depth_maps[i].astype(np.float32)
            d_max = minmax_list[i, 0]
            d_min = minmax_list[i, 1]

            if d_max - d_min == 0:
                print('Bug')

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
    def ProcessRGBs(rgbs):
        imagesN = ProcessingRGBDDReal._NormalizeRGB(rgbs)
        imagesS = ProcessingRGBDDReal._StandardizeRGB(imagesN)

        return imagesN, imagesS

    @staticmethod
    def ProcessDepths(depths_hr, depths_lr, low: float, high: float):
        # 1. Generate the masks and clip both HR and LR depth with the same range
        masks, depth_mapsC = ProcessingRGBDDReal._GenerateDepthMaskBatch(depths_hr, low, high)
        masksLR, depth_mapsLR_C = ProcessingRGBDDReal._GenerateDepthMaskBatch(depths_lr, low, high)

        # 2. Normalize the LR depth and keep its min/max (this is what is
        #    actually available in the real-world scenario)
        depth_mapsLR_N, minmax_list = ProcessingRGBDDReal._NormalizeDepth(depth_mapsLR_C)

        # 3. Normalize the HR ground truth with the LR min/max so training
        #    targets and de-normalization are consistent with the LR input
        # Why it's correct. The rule for any normalization is: the statistics you normalize the target with must be recoverable at inference time, because that's what you'll use to denormalize the prediction. At test time you have only the LR depth, so per-sample LR min/max is the only per-sample statistic available. If you normalized GT with its own min/max, training would work, but at inference you'd have no way to map the network's [0,1] output back to meters — the whole pipeline would be broken. So:
        depth_mapsN = ProcessingRGBDDReal._NormalizeDepthWithMinMax(depth_mapsC, minmax_list)

        return masks, depth_mapsC, depth_mapsN, masksLR, depth_mapsLR_C, depth_mapsLR_N, minmax_list

    @staticmethod
    def _LoadAllImages(paths):
        rgb_images = []
        depth_hr_images = []
        depth_lr_images = []

        for depth_hr_path, rgb_path, depth_lr_path in paths:
            # Load RGB (H, W, 3)
            rgb = np.array(Image.open(rgb_path).convert("RGB"))

            # Load HR Depth (H, W) in meters
            depth_hr = np.array(Image.open(depth_hr_path)) / 1000.0

            # Load real LR Depth from the phone ToF sensor (H/2, W/2) in meters
            depth_lr = np.array(Image.open(depth_lr_path)) / 1000.0

            rgb_images.append(rgb)
            depth_hr_images.append(depth_hr)
            depth_lr_images.append(depth_lr)

        return np.stack(rgb_images), np.stack(depth_hr_images), np.stack(depth_lr_images)

    @staticmethod
    def _InitDataDict(path: str, N: int, prefix: str = 'train'):
        HR_H, HR_W = ProcessingRGBDDReal.HR_H, ProcessingRGBDDReal.HR_W
        LR_H, LR_W = ProcessingRGBDDReal.LR_H, ProcessingRGBDDReal.LR_W
        return {
                "imagesT": open_memmap(path + f"{prefix}_images_split.npy", 'w+', np.uint8, (N, 3, HR_H, HR_W)),
                "imagesN": open_memmap(path + f"{prefix}_images_norm_split.npy", 'w+', np.float32, (N, 3, HR_H, HR_W)),
                "imagesS": open_memmap(path + f"{prefix}_images_stand_split.npy", 'w+', np.float32, (N, 3, HR_H, HR_W)),

                "depthT": open_memmap(path + f"{prefix}_depths_split.npy", 'w+', np.float32, (N, HR_H, HR_W)),
                "depth_mapsC": open_memmap(path + f"{prefix}_depths_clipped_split.npy", 'w+', np.float32, (N, HR_H, HR_W)),
                "depth_mapsN": open_memmap(path + f"{prefix}_depths_norm_split.npy", 'w+', np.float32, (N, HR_H, HR_W)),

                "depthLR_T": open_memmap(path + f"{prefix}_depths_lr_split.npy", 'w+', np.float32, (N, LR_H, LR_W)),
                "depthLR_C": open_memmap(path + f"{prefix}_depths_lr_clipped_split.npy", 'w+', np.float32, (N, LR_H, LR_W)),
                "depthLR_N": open_memmap(path + f"{prefix}_depths_lr_norm_split.npy", 'w+', np.float32, (N, LR_H, LR_W)),

                "masks": open_memmap(path + f"{prefix}_mask_split.npy", 'w+', bool, (N, HR_H, HR_W)),
                "masksLR": open_memmap(path + f"{prefix}_mask_lr_split.npy", 'w+', bool, (N, LR_H, LR_W)),
                "minmax_list": open_memmap(path + f"{prefix}_minmax_split.npy", 'w+', np.float32, (N, 2)),
            }

    @staticmethod
    def ProcessData(pairs: list[tuple], path: str, batch_size: int, prefix: str = 'train'):
        # 1. Count input examples and init the required np arrays on disk
        N = len(pairs)
        print('Processing ' + prefix + '. Total examples: ', N)
        collect = ProcessingRGBDDReal._InitDataDict(path, N, prefix)

        # 2. For each batch save the data in the collect files
        for start in range(0, N, batch_size):
            # 2.1. pick the batch
            end = min(start + batch_size, N)

            batch = pairs[start:end]
            paths, lows, highs = zip(*batch)

            # 2.2. Load and process images
            rgbs, depths_hr, depths_lr = ProcessingRGBDDReal._LoadAllImages(paths)
            rgbs = np.transpose(rgbs, (0, -1, 1, 2))
            imagesN, imagesS = ProcessingRGBDDReal.ProcessRGBs(rgbs)
            masks, depth_mapsC, depth_mapsN, masksLR, depth_mapsLR_C, depth_mapsLR_N, minmax_list = \
                ProcessingRGBDDReal.ProcessDepths(depths_hr, depths_lr, lows, highs)

            # 2.3. Save data
            collect['imagesT'][start:end] = rgbs
            collect['imagesN'][start:end] = imagesN
            collect['imagesS'][start:end] = imagesS

            collect['depthT'][start:end] = depths_hr
            collect['depth_mapsC'][start:end] = depth_mapsC
            collect['depth_mapsN'][start:end] = depth_mapsN

            collect['depthLR_T'][start:end] = depths_lr
            collect['depthLR_C'][start:end] = depth_mapsLR_C
            collect['depthLR_N'][start:end] = depth_mapsLR_N

            collect['masks'][start:end] = masks
            collect['masksLR'][start:end] = masksLR
            collect['minmax_list'][start:end] = minmax_list

    @staticmethod
    def GenerateNPYFiles(batch_size: int = 32):
        # 1. Init example paths
        model_train = PathManager.GetBasePath() + 'RGBDD-Full/models/models_train'
        model_test = PathManager.GetBasePath() + 'RGBDD-Full/models/models_test'
        plants_train = PathManager.GetBasePath() + 'RGBDD-Full/plants/plants_train'
        plants_test = PathManager.GetBasePath() + 'RGBDD-Full/plants/plants_test'
        portraits_train = PathManager.GetBasePath() + 'RGBDD-Full/portraits/portraits_train'
        portraits_test = PathManager.GetBasePath() + 'RGBDD-Full/portraits/portraits_test'

        model_train_pairs = ProcessingRGBDDReal._LoadPairPaths(model_train)
        model_test_pairs = ProcessingRGBDDReal._LoadPairPaths(model_test)
        plants_train_pairs = ProcessingRGBDDReal._LoadPairPaths(plants_train)
        plants_test_pairs = ProcessingRGBDDReal._LoadPairPaths(plants_test)
        portraits_train_pairs = ProcessingRGBDDReal._LoadPairPaths(portraits_train)
        portraits_test_pairs = ProcessingRGBDDReal._LoadPairPaths(portraits_test)

        # 2. Init the output path
        path = PathManager.GetBasePath() + BenchmarkType.RGBDDReal.name + '/'
        DirectoryHelper.ResetFolder(path)

        # 3. Merge training examples
        train_pairs = []
        test_pairs = []
        for i, low, high in [(model_train_pairs, 0.6, 3), (portraits_train_pairs, 1, 5), (plants_train_pairs, 0.6, 1.5)]:
            for v in i.values():
                train_pairs.append(
                    (
                        v, low, high
                    )
                )

        # 4. Merge testing examples
        for i, low, high in [(model_test_pairs, 0.6, 3), (portraits_test_pairs, 1, 5), (plants_test_pairs, 0.6, 1.5)]:
            for v in i.values():
                test_pairs.append(
                    (
                        v, low, high
                    )
                )

        train_count = len(train_pairs)
        test_count  = len(test_pairs)

        print("Train samples:", train_count)
        print("Test samples:", test_count)

        # 5. Process data
        ProcessingRGBDDReal.ProcessData(train_pairs, path, batch_size, 'train')
        ProcessingRGBDDReal.ProcessData(test_pairs, path, batch_size, 'test')