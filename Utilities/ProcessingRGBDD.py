from Models.BenchmarkType import BenchmarkType
from Utilities.DirectoryHelper import DirectoryHelper
from Utilities.PathManager import PathManager
import numpy as np
import os
from PIL import Image
from numpy.lib.format import open_memmap

class ProcessingRGBDD:
    @staticmethod
    def _LoadPairPaths(path: str):
        pairs = {}
        for index, obj in enumerate(os.walk(path)):
            root, _, files = obj
            depth = None
            rgb = None
            for file in files:
                if file.endswith('HR_gt.png'):
                    depth = os.path.join(root, file)
                if file.endswith('RGB.jpg'):
                    rgb = os.path.join(root, file)

            if rgb is None or depth is None:
                print(root)
            else:
                pairs[index] = (depth, rgb)

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
        imagesN = ProcessingRGBDD._NormalizeRGB(rgbs)
        imagesS = ProcessingRGBDD._StandardizeRGB(imagesN)

        return imagesN, imagesS

    @staticmethod
    def ProcessDepths(depths, low: float, high: float):
        # Generate the mask and Clip depth
        masks, depth_mapsC = ProcessingRGBDD._GenerateDepthMaskBatch(depths, low, high)

        # Generate the normalized version, and min, max
        depth_mapsN, minmax_list = ProcessingRGBDD._NormalizeDepth(depth_mapsC)

        return masks, depth_mapsC, depth_mapsN, minmax_list

    @staticmethod
    def _LoadAllImages(paths):
        rgb_images = []
        depth_images = []

        for depth_path, rgb_path in paths:
            # Load RGB (H, W, 3)
            rgb = np.array(Image.open(rgb_path).convert("RGB"))

            # Load Depth (H, W)
            depth = np.array(Image.open(depth_path)) / 1000.0

            rgb_images.append(rgb)
            depth_images.append(depth)

        return np.stack(rgb_images), np.stack(depth_images)

    @staticmethod
    def _InitDataDict(path: str, N: int, prefix: str = 'train'):
        return {
                "imagesT": open_memmap(path + f"{prefix}_images_split.npy", 'w+', np.uint8, (N, 3, 384, 512)),
                "imagesN": open_memmap(path + f"{prefix}_images_norm_split.npy", 'w+', np.float32, (N, 3, 384, 512)),
                "imagesS": open_memmap(path + f"{prefix}_images_stand_split.npy", 'w+', np.float32, (N, 3, 384, 512)),

                "depthT": open_memmap(path + f"{prefix}_depths_split.npy", 'w+', np.float32, (N, 384, 512)),
                "depth_mapsC": open_memmap(path + f"{prefix}_depths_clipped_split.npy", 'w+', np.float32, (N, 384, 512)),
                "depth_mapsN": open_memmap(path + f"{prefix}_depths_norm_split.npy", 'w+', np.float32, (N, 384, 512)),

                "masks": open_memmap(path + f"{prefix}_mask_split.npy", 'w+', bool, (N, 384, 512)),
                "minmax_list": open_memmap(path + f"{prefix}_minmax_split.npy", 'w+', np.float32, (N, 2)),
            }

    @staticmethod
    def ProcessData(pairs: list[tuple], path: str, batch_size: int, prefix: str = 'train'):
        # 1. Count iput exmaples and init the required np arrays on disk
        N = len(pairs)
        print('Processing ' + prefix + '. Total examples: ', N)
        collect = ProcessingRGBDD._InitDataDict(path, N, prefix)

        # 2. For each batch save the data in the collect files
        for start in range(0, N, batch_size):
            # 2.1. pick the batch
            end = min(start + batch_size, N)

            batch = pairs[start:end]
            paths, lows, highs = zip(*batch)

            # 2.2. Load and process images
            rgbs, depths = ProcessingRGBDD._LoadAllImages(paths)
            rgbs = np.transpose(rgbs, (0, -1, 1 ,2))
            imagesN, imagesS = ProcessingRGBDD.ProcessRGBs(rgbs)
            masks, depth_mapsC, depth_mapsN, minmax_list = ProcessingRGBDD.ProcessDepths(depths, lows, highs)

            # 2.3. Save data
            collect['imagesT'][start:end] = rgbs
            collect['depthT'][start:end] = depths
            collect['imagesN'][start:end] = imagesN
            collect['imagesS'][start:end] = imagesS
            collect['masks'][start:end] = masks
            collect['depth_mapsC'][start:end] = depth_mapsC
            collect['depth_mapsN'][start:end] = depth_mapsN
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
        
        model_train_pairs = ProcessingRGBDD._LoadPairPaths(model_train)
        model_test_pairs = ProcessingRGBDD._LoadPairPaths(model_test)
        plants_train_pairs = ProcessingRGBDD._LoadPairPaths(plants_train)
        plants_test_pairs = ProcessingRGBDD._LoadPairPaths(plants_test)
        portraits_train_pairs = ProcessingRGBDD._LoadPairPaths(portraits_train)
        portraits_test_pairs = ProcessingRGBDD._LoadPairPaths(portraits_test)

        # 2. Init the output path
        path = PathManager.GetBasePath() + BenchmarkType.RGBDD.name + '/'
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
        ProcessingRGBDD.ProcessData(train_pairs, path, batch_size, 'train')
        ProcessingRGBDD.ProcessData(test_pairs, path, batch_size, 'test')
