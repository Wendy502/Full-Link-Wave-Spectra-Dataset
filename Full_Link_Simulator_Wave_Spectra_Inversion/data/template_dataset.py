"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import numpy as np
import xarray as xr
import cv2
#import matplotlib.pyplot as plt
from data.base_dataset import BaseDataset, get_transform,get_params,get_transform_wave
from data.image_folder import make_dataset
from PIL import Image
import scipy.io as sio
import h5py
import os
import matplotlib.pyplot as plt
import torch
import scipy.io as scio
import random
import cv2
from scipy.io import savemat
class TemplateDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.set_defaults(max_dataset_size=100000000, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        print(self.AB_paths)
        print(self.dir_AB)
        assert (self.opt.load_size >= self.opt.crop_size)
        #self.image_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        #self.transform = get_transform(opt)
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        # read a image given a random integer index
        # print(AB_path)
        sar_img = 0  #是否SAR图像
        Height = 0 #是否高度场
        import_SIZE = 256
        nor_option=1 #是否对输入进行归一化
        AB_path = self.AB_paths[index]
        AB = h5py.File(AB_path, 'r')
        # AB = scio.loadmat(AB_path)
        if sar_img: #如果是真实数据训练
            A_1 = np.array(AB['Sar_Area'][:]).astype(np.float32)  # 读取SAR图像
            # A_1 = cv2.resize(A_1,(import_SIZE,import_SIZE),interpolation=cv2.INTER_LINEAR)
        else:
            # A_1 = np.array(AB['simu_spec_vv'][:]).astype(np.float32)  # 读取VV
            A_1 = np.array(AB['SAR_Spectra'][:]).astype(np.float32)  # 读取功率谱


        if Height:
            B_1 = np.array(AB['Height_Area'][:]).astype(np.float32)  #读取高度场
            # B_1 = cv2.resize(B_1,(import_SIZE,import_SIZE),interpolation=cv2.INTER_LINEAR)
            # 输出必须归一化
            B_1_nor = cv2.normalize(B_1, None, 0, 1, cv2.NORM_MINMAX) #对输出高度场0-1归一化
            A_3 = np.array(AB['incidenceAngle'][:]).astype(np.float32)  # 读取入射角
            A_3 = np.mean(np.mean(A_3)) * np.ones((import_SIZE,import_SIZE), dtype=np.float32)
        else:
            #B_1 = np.array(AB['era_spec'][:]).astype(np.float32)
            # auto-detect wave spectrum key: Wave_Spectra (simulated) or Buoy_spec (observed)
            if 'Wave_Spectra' in AB:
                B_1 = np.array(AB['Wave_Spectra'][:]).astype(np.float32)
            elif 'Buoy_spec' in AB:
                B_1 = np.array(AB['Buoy_spec'][:]).astype(np.float32)
            else:
                raise KeyError('No wave spectrum key (Wave_Spectra or Buoy_spec) found in file')
            # B_1 = 1/2 * (B_1 + np.rot90(np.rot90(B_1)))    #转换至双边谱
            B_1 = 1/2 * (B_1 + np.rot90(np.rot90(B_1)))  # 转换至双边谱
            B_1_nor = np.log10(B_1 + 1) / 4         #波浪谱归一化
            # B_1_nor = cv2.normalize(B_1, None, 0, 1, cv2.NORM_MINMAX)
            A_3 = np.array(AB['incidenceAngle_Center'][:]).astype(np.float32)  # 读取入射角
            # A_3 = np.array(AB['incidenceAngle_Matrix'][:]).astype(np.float32)  # 读取入射角
            A_3 = np.mean(np.mean(A_3)) * np.ones((import_SIZE,import_SIZE), dtype=np.float32)

        # 斜距
        nearRange = np.array(AB['nearRange'][:]).astype(np.float32)
        farRange = np.array(AB['farRange'][:]).astype(np.float32)
        Range = (nearRange + farRange)/2
        A_2 = Range * np.ones((import_SIZE, import_SIZE), dtype=np.float32)
        # Velocity = np.array(AB['sat_Velocity'][:]).astype(np.float32)
        # A_2 = Range / Velocity * np.ones((import_SIZE,import_SIZE), dtype=np.float32)
        #A_1_nor=(A_1)#/150*1
        #B_1_nor=(B_1)/50-0.5
        k_grid_max = 0.25
        dkx_grid = np.linspace(-k_grid_max, k_grid_max, import_SIZE)
        kx_grid, ky_grid = np.meshgrid(dkx_grid, dkx_grid)
        k_grid = (np.sqrt(kx_grid * kx_grid + ky_grid * ky_grid))
        ##### 以下*k_grid来缩小海浪谱极值的方法不可靠
        #B_11 = B_11 *k_grid
        #print('max_b1',np.max(B_1))
        #A_1_nor = cv2.normalize(A_1, None, 0, 1, cv2.NORM_MINMAX)
        #即：每个海浪谱都是归一化0-1之间
        if nor_option==1:
            A_1 = cv2.normalize(A_1, None, 0, 1, cv2.NORM_MINMAX)   #0-1归一化
            # A_1 = A_1 / 1000
            # A_1 = np.log10(A_1 + 1) / 3     #对数归一化
            # 斜距归一化，依据斜距WV1模式斜距取值范围通常是
            min_inc=20
            A_3=(A_3-min_inc)/40
            min_Range = 7.0*1e5
            max_Range = 9.0*1e5
            A_2 = (A_2 - min_Range) / max_Range
            # min_alpha = 90
            # max_alpha = 120
            # A_2 = (A_2 - min_alpha) / max_alpha
        # real SAR
        A_1=A_1.reshape(import_SIZE,import_SIZE,-1)
        A_2=A_2.reshape(import_SIZE,import_SIZE,-1)
        A_3=A_3.reshape(import_SIZE,import_SIZE,-1)
        # 图像谱，平均斜距，入射角矩阵
        A=np.concatenate((A_1,A_2,A_3),axis=2)
        # A = np.concatenate((A_1, A_3), axis=2)
        # A = A_1
        B_1_nor = B_1_nor.reshape(import_SIZE, import_SIZE,-1)
        B=B_1_nor
        ### 让数据正态分布，使其便于训练。
        transform_params = get_params(self.opt, A.shape)

        A_transform = get_transform_wave(self.opt, transform_params)
        # not use normalize
        B_transform = get_transform_wave(self.opt, transform_params, grayscale=True)

        A = A_transform(A) #应用归一化
        B = B_transform(B)
        B=np.nan_to_num(B)
        # print(A.shape,'single')
        # if 1:
        #     #print(B_1)
        #     plt.contourf(B_1)
        #     plt.colorbar()
        #     plt.show()
        # plt.contourf(A_1)
        # plt.colorbar()
        # plt.show()
        # plt.contourf(A_3)
        # plt.show()
        # A_1_nor = (A_1-0.5) / 150 * 1
        # B_1_nor = (B_1) / 800 * 2
        # if np.max(B_1_nor)>1:
        #    B_1_nor=cv2.normalize(B_1, None, 0, 1, cv2.NORM_MINMAX)
        # cv2.normalize(A_2, A_2, 0, 1, cv2.NORM_MINMAX)
        # max_inc=60
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images."""

        return len(self.AB_paths)
