import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F


import random
from basicsr.utils import img2tensor, tensor2img

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PairedSROnlineTxtDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()
        self.args = args
        self.split = split
        if split == 'train':

            set_random_seed(self.args.seed)
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((args.resolution, args.resolution)),
                transforms.RandomHorizontalFlip(),
            ])

            self.gt_refined_list_2x = []
            assert len(args.train_dataset_txt_paths_list_gt_refined_2x) == len(args.dataset_prob_paths_list)
            for idx_dataset in range(len(args.train_dataset_txt_paths_list_gt_refined_2x)):
                with open(args.train_dataset_txt_paths_list_gt_refined_2x[idx_dataset], 'r') as f:
                    dataset_list = [line.strip() for line in f.readlines()]
                    for idx_ratio in range(args.dataset_prob_paths_list[idx_dataset]):
                        gt_length = len(self.gt_refined_list_2x)
                        self.gt_refined_list_2x += dataset_list
                        print(f'=====> append {len(self.gt_refined_list_2x) - gt_length} data.')

            self.gt_refined_list_3x = []
            assert len(args.train_dataset_txt_paths_list_gt_refined_3x) == len(args.dataset_prob_paths_list)
            for idx_dataset in range(len(args.train_dataset_txt_paths_list_gt_refined_3x)):
                with open(args.train_dataset_txt_paths_list_gt_refined_3x[idx_dataset], 'r') as f:
                    dataset_list = [line.strip() for line in f.readlines()]
                    for idx_ratio in range(args.dataset_prob_paths_list[idx_dataset]):
                        gt_length = len(self.gt_refined_list_3x)
                        self.gt_refined_list_3x += dataset_list
                        print(f'=====> append {len(self.gt_refined_list_3x) - gt_length} data.')

            self.gt_refined_list_4x = []
            assert len(args.train_dataset_txt_paths_list_gt_refined_4x) == len(args.dataset_prob_paths_list)
            for idx_dataset in range(len(args.train_dataset_txt_paths_list_gt_refined_4x)):
                with open(args.train_dataset_txt_paths_list_gt_refined_4x[idx_dataset], 'r') as f:
                    dataset_list = [line.strip() for line in f.readlines()]
                    for idx_ratio in range(args.dataset_prob_paths_list[idx_dataset]):
                        gt_length = len(self.gt_refined_list_4x)
                        self.gt_refined_list_4x += dataset_list
                        print(f'=====> append {len(self.gt_refined_list_4x) - gt_length} data.')

            self.gt_list = []
            assert len(args.train_dataset_txt_paths_list_gt) == len(args.dataset_prob_paths_list)
            for idx_dataset in range(len(args.train_dataset_txt_paths_list_gt)):
                with open(args.train_dataset_txt_paths_list_gt[idx_dataset], 'r') as f:
                    dataset_list = [line.strip() for line in f.readlines()]
                    for idx_ratio in range(args.dataset_prob_paths_list[idx_dataset]):
                        gt_length = len(self.gt_list)
                        self.gt_list += dataset_list
                        print(f'=====> append {len(self.gt_list) - gt_length} data.')

            self.lq_list = []
            assert len(args.train_dataset_txt_paths_list_lq) == len(args.dataset_prob_paths_list)
            for idx_dataset in range(len(args.train_dataset_txt_paths_list_lq)):
                with open(args.train_dataset_txt_paths_list_lq[idx_dataset], 'r') as f:
                    dataset_list = [line.strip() for line in f.readlines()]
                    for idx_ratio in range(args.dataset_prob_paths_list[idx_dataset]):
                        gt_length = len(self.lq_list)
                        self.lq_list += dataset_list
                        print(f'=====> append {len(self.lq_list) - gt_length} data.')
        elif split == 'test':
            self.gt_list = []
            for idx_dataset in range(len(args.test_dataset_txt_paths_list_gt)):
                with open(args.test_dataset_txt_paths_list_gt[idx_dataset], 'r') as f:
                    dataset_list = [line.strip() for line in f.readlines()]
                    for idx_ratio in range(args.dataset_prob_paths_list[idx_dataset]):
                        gt_length = len(self.gt_list)
                        self.gt_list += dataset_list
                        print(f'=====> append {len(self.gt_list) - gt_length} data.')

            self.lq_list = []
            for idx_dataset in range(len(args.test_dataset_txt_paths_list_lq)):
                with open(args.test_dataset_txt_paths_list_lq[idx_dataset], 'r') as f:
                    dataset_list = [line.strip() for line in f.readlines()]
                    for idx_ratio in range(args.dataset_prob_paths_list[idx_dataset]):
                        gt_length = len(self.lq_list)
                        self.lq_list += dataset_list
                        print(f'=====> append {len(self.lq_list) - gt_length} data.')


    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'train':
            gt_img = Image.open(self.gt_list[idx]).convert('RGB')
            gt_refined_img_2x = Image.open(self.gt_refined_list_2x[idx]).convert('RGB')
            gt_refined_img_3x = Image.open(self.gt_refined_list_3x[idx]).convert('RGB')
            gt_refined_img_4x = Image.open(self.gt_refined_list_4x[idx]).convert('RGB')
            lq_img = Image.open(self.lq_list[idx]).convert('RGB')

            crop_size = (self.args.resolution, self.args.resolution)
            i, j, h, w = transforms.RandomCrop.get_params(gt_img, output_size=crop_size)

            gt_img = F.crop(gt_img, i, j, h, w)
            gt_refined_img_2x = F.crop(gt_refined_img_2x, i, j, h, w)
            gt_refined_img_3x = F.crop(gt_refined_img_3x, i, j, h, w)
            gt_refined_img_4x = F.crop(gt_refined_img_4x, i, j, h, w)
            lq_img = F.crop(lq_img, i, j, h, w)

            if random.random() < 0.5:
                gt_img = F.hflip(gt_img)
                gt_refined_img_2x = F.hflip(gt_refined_img_2x)
                gt_refined_img_3x = F.hflip(gt_refined_img_3x)
                gt_refined_img_4x = F.hflip(gt_refined_img_4x)
                lq_img = F.hflip(lq_img)

            gt_img = np.asarray(gt_img) / 255.
            gt_refined_img_2x = np.asarray(gt_refined_img_2x) / 255.
            gt_refined_img_3x = np.asarray(gt_refined_img_3x) / 255.
            gt_refined_img_4x = np.asarray(gt_refined_img_4x) / 255.
            lq_img = np.asarray(lq_img) / 255.

        
            output_t = img2tensor([gt_img], bgr2rgb=False, float32=True)[0].unsqueeze(0) 
            output_refined_t_2x = img2tensor([gt_refined_img_2x], bgr2rgb=False, float32=True)[0].unsqueeze(0) 
            output_refined_t_3x = img2tensor([gt_refined_img_3x], bgr2rgb=False, float32=True)[0].unsqueeze(0) 
            output_refined_t_4x = img2tensor([gt_refined_img_4x], bgr2rgb=False, float32=True)[0].unsqueeze(0) 
            img_t = img2tensor([lq_img], bgr2rgb=False, float32=True)[0].unsqueeze(0) 


            output_t, img_t, output_refined_t_2x = output_t.squeeze(0), img_t.squeeze(0), output_refined_t_2x.squeeze(0)
            output_refined_t_3x = output_refined_t_3x.squeeze(0)
            output_refined_t_4x = output_refined_t_4x.squeeze(0)


            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])
            output_refined_t_2x = F.normalize(output_refined_t_2x, mean=[0.5], std=[0.5])
            output_refined_t_3x = F.normalize(output_refined_t_3x, mean=[0.5], std=[0.5])
            output_refined_t_4x = F.normalize(output_refined_t_4x, mean=[0.5], std=[0.5])

            example = {}
            example["output_pixel_values"] = output_t
            example["output_refiend_pixel_values_2x"] = output_refined_t_2x
            example["output_refiend_pixel_values_3x"] = output_refined_t_3x
            example["output_refiend_pixel_values_4x"] = output_refined_t_4x
            example["conditioning_pixel_values"] = img_t
            example["name"] = self.lq_list[idx]

            return example

        else:
            gt_img = Image.open(self.gt_list[idx]).convert('RGB')
            lq_img = Image.open(self.lq_list[idx]).convert('RGB')
            lq_img = lq_img.resize((gt_img.width, gt_img.height), Image.BICUBIC)
            lq_img = np.asarray(lq_img) / 255.
            gt_img = np.asarray(gt_img) / 255.

            output_t = img2tensor([gt_img], bgr2rgb=False, float32=True)[0].unsqueeze(0) 
            img_t = img2tensor([lq_img], bgr2rgb=False, float32=True)[0].unsqueeze(0) 

            output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)

            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t
            }
            
            return example

            
        


