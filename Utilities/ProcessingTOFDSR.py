from Models.BenchmarkType import BenchmarkType
from Utilities.DirectoryHelper import DirectoryHelper
from Utilities.PathManager import PathManager
import numpy as np
from PIL import Image
from numpy.lib.format import open_memmap

class ProcessingTOFDSR:
    TRAIN_FILE = PathManager.GetBasePath() + 'TOFDSR/TOFDSR_Train.txt'
    TEST_FILE = PathManager.GetBasePath() + 'TOFDSR/TOFDSR_Test.txt'

    BASE = PathManager.GetBasePath() + 'TOFDSR' # No backslash

    @staticmethod
    def _GetPairs(base_path: str, base: str):
        pairs = []
        with open(base_path, "r") as f:
            for index, line in enumerate(f):
                line = line.strip()          # remove newline and extra spaces
                parts = line.split(",")      # split by comma
                
                pairs.append((base + parts[1].lstrip('TOFDC_split'), base + parts[0].lstrip('TOFDC_split')))

        return pairs

    @staticmethod
    def _LoadPaths():
        return ProcessingTOFDSR._GetPairs(ProcessingTOFDSR.TRAIN_FILE, ProcessingTOFDSR.BASE), ProcessingTOFDSR._GetPairs(ProcessingTOFDSR.TEST_FILE, ProcessingTOFDSR.BASE)

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
        imagesT_mm = open_memmap(save_path + prefix + '_images_split.npy', mode='w+', dtype=np.uint8, shape=(N, 3, 384, 512))
        imagesN_mm = open_memmap(save_path + prefix + '_images_norm_split.npy', mode='w+', dtype=np.float32, shape=(N, 3, 384, 512))
        imagesS_mm = open_memmap(save_path + prefix + '_images_stand_split.npy', mode='w+', dtype=np.float32, shape=(N, 3, 384, 512))

        depthT_mm  = open_memmap(save_path + prefix + '_depths_split.npy', mode='w+', dtype=np.float32, shape=(N, 384, 512))
        depthC_mm  = open_memmap(save_path + prefix + '_depths_clipped_split.npy', mode='w+', dtype=np.float32, shape=(N, 384, 512))
        depthN_mm  = open_memmap(save_path + prefix + '_depths_norm_split.npy', mode='w+', dtype=np.float32, shape=(N, 384, 512))

        mask_mm    = open_memmap(save_path + prefix + '_mask_split.npy', mode='w+', dtype=bool, shape=(N, 384, 512))
        minmax_mm  = open_memmap(save_path + prefix + '_minmax_split.npy', mode='w+', dtype=np.float32, shape=(N, 2))

        # 2. Process the batches
        print("Processing batches...")
        for start in range(0, N, batch_size):
            # 2.1. pick the batch and load data
            end = min(start + batch_size, N)
            paths = pairs[start:end]
            imagesT, depth_mapsT = ProcessingTOFDSR._LoadAllImages(paths)

            # 2.2. Store the base image and depth            
            imagesT = np.transpose(imagesT, (0, -1, 1 ,2))
            imagesT_mm[start:end] = imagesT
            depthT_mm[start:end]  = depth_mapsT

            # 2.3. Normalize RGB using imagenet weights
            imagesN = ProcessingTOFDSR._NormalizeRGB(imagesT)
            imagesS = ProcessingTOFDSR._StandardizeRGB(imagesN)
            imagesN_mm[start:end] = imagesN
            imagesS_mm[start:end] = imagesS

            # 2.4. Generate a mask for depth pixel out of range
            masks = ProcessingTOFDSR._GenerateDepthMaskBatch(depth_mapsT)
            mask_mm[start:end] = masks

            # 2.5. Clip the depths between 0.1 and 10 m
            depth_mapsC = np.clip(depth_mapsT, 0.1, 6.0)
            depthC_mm[start:end] = depth_mapsC

            # 2.6. Generate min max normalized verison of the depth, and min max maps
            depth_mapsN, minmax_list = ProcessingTOFDSR._NormalizeDepth(depth_mapsC)
            depthN_mm[start:end] = depth_mapsN
            minmax_mm[start:end] = minmax_list

    @staticmethod
    def GenerateNPYFiles(batch_size: int = 32):
        # 1. Load data paths
        ProcessingTOFDSR.TRAIN_FILE = PathManager.GetBasePath() + 'TOFDSR/TOFDSR_Train.txt'
        ProcessingTOFDSR.TEST_FILE = PathManager.GetBasePath() + 'TOFDSR/TOFDSR_Test.txt'
        ProcessingTOFDSR.BASE = PathManager.GetBasePath() + 'TOFDSR' # No backslash
        train_pairs, test_pairs = ProcessingTOFDSR._LoadPaths()

        # 2. Create the output path
        path = PathManager.GetBasePath() + BenchmarkType.TOFDSRD.name + '/'
        DirectoryHelper.ResetFolder(path)

        # 3. Process the data
        ProcessingTOFDSR.ProcessBatches(train_pairs, 'train', path, batch_size)
        ProcessingTOFDSR.ProcessBatches(test_pairs, 'test', path, batch_size)