from Models.BenchmarkType import BenchmarkType
from Utilities.DirectoryHelper import DirectoryHelper
from Utilities.PathManager import PathManager
import h5py
import numpy as np
from numpy.lib.format import open_memmap

class ProcessingNYUMat:
    @staticmethod
    def _LoadFromH5(path: str):
        file = h5py.File(path)

        images = file['images']
        depth_maps = file['depths']
        instances = file['instances']
        labels = file['labels']

        return images, depth_maps

    @staticmethod
    def _Transpose(images, depth_maps):
        imgs = np.array(images)   
        dpths = np.array(depth_maps)  

        imgs = np.transpose(imgs, (0, 1, 3 ,2))  
        dpths = np.transpose(dpths, (0, 2, 1))

        return imgs, dpths


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
    def _GenerateDepthMaskBatch(depth_maps, min_depth=0.1, max_depth=10.0):
        mask = (depth_maps >= min_depth) & (depth_maps <= max_depth)
        return mask

    @staticmethod
    def ProcessBatches(images, depth_maps, batch_size: int, save_path: str, prefix: str = 'train'):
        # 1. Create memmaps (disk-backed arrays)
        N = images.shape[0]
        imagesT_mm = open_memmap(save_path + prefix + '_images_split.npy', mode='w+', dtype=np.uint8, shape=(N, 3, 480, 640))
        imagesN_mm = open_memmap(save_path + prefix + '_images_norm_split.npy', mode='w+', dtype=np.float32, shape=(N, 3, 480, 640))
        imagesS_mm = open_memmap(save_path + prefix + '_images_stand_split.npy', mode='w+', dtype=np.float32, shape=(N, 3, 480, 640))

        depthT_mm  = open_memmap(save_path + prefix + '_depths_split.npy', mode='w+', dtype=np.float32, shape=(N, 480, 640))
        depthC_mm  = open_memmap(save_path + prefix + '_depths_clipped_split.npy', mode='w+', dtype=np.float32, shape=(N, 480, 640))
        depthN_mm  = open_memmap(save_path + prefix + '_depths_norm_split.npy', mode='w+', dtype=np.float32, shape=(N, 480, 640))

        mask_mm    = open_memmap(save_path + prefix + '_mask_split.npy', mode='w+', dtype=bool, shape=(N, 480, 640))
        minmax_mm  = open_memmap(save_path + prefix + '_minmax_split.npy', mode='w+', dtype=np.float32, shape=(N, 2))

        # 2. Start batch processing
        print("Processing batches...")
        for start in range(0, N, batch_size):
            # 2.1. pick the batch
            end = min(start + batch_size, N)

            img_batch = images[start:end]     # still lazy read
            d_batch   = depth_maps[start:end]

            # 2.2. Change the W, H format for H, W
            imagesT, depth_mapsT = ProcessingNYUMat._Transpose(img_batch, d_batch)
            imagesT_mm[start:end] = imagesT
            depthT_mm[start:end]  = depth_mapsT

            # 2.3. Normalize RGB using imagenet weights
            imagesN = ProcessingNYUMat._NormalizeRGB(imagesT)
            imagesS = ProcessingNYUMat._StandardizeRGB(imagesN)
            imagesN_mm[start:end] = imagesN
            imagesS_mm[start:end] = imagesS

            # 2.4. Generate a mask for depth pixel out of range
            masks = ProcessingNYUMat._GenerateDepthMaskBatch(depth_mapsT)
            mask_mm[start:end] = masks

            # 2.5. Clip the depths between 0.1 and 10 m
            depth_mapsC = np.clip(depth_mapsT, 0.1, 10.0) # For NYU depths outside this range are noise
            depthC_mm[start:end] = depth_mapsC

            # 2.6. Generate min max normalized verison of the depth, and min max maps
            depth_mapsN, minmax_list = ProcessingNYUMat._NormalizeDepth(depth_mapsC) # removing noise before normalizing
            depthN_mm[start:end] = depth_mapsN
            minmax_mm[start:end] = minmax_list


    @staticmethod
    def GenerateNPYFiles(batch_size: int = 32):
        print('Loading data')
        path = PathManager.GetBasePath() + 'nyu_depth_v2_labeled.mat'
        # 1. Load the data
        images, depth_maps = ProcessingNYUMat._LoadFromH5(path)

        N = images.shape[0]
        print("Dataset size:", N)

        # 2. Create the save path
        save_path = PathManager.GetBasePath() + BenchmarkType.NYUV2.name + '/'
        DirectoryHelper.ResetFolder(save_path)

        # 3. Split the data into train and test
        train_ids = list(range(0, 1000))
        test_ids = list(range(1000, 1449))

        train_images = images[train_ids]
        train_depths = depth_maps[train_ids]

        test_images = images[test_ids]
        test_depths = depth_maps[test_ids]

        images = None
        depth_maps = None

        # 4. Process data in batches
        ProcessingNYUMat.ProcessBatches(train_images, train_depths, batch_size, save_path, 'train')
        ProcessingNYUMat.ProcessBatches(test_images, test_depths, batch_size, save_path, 'test')