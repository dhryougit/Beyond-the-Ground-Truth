import os
import sys
import glob
import argparse
import logging
import time
import tempfile
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image

from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage

from tqdm.auto import tqdm

# pyiqa for image quality assessment
import pyiqa

# Accelerate for multi-GPU distributed evaluation
from accelerate import Accelerator
from accelerate.utils import set_seed

from arch.ORNet_arch import ORNet

##############################################################################
# Utility functions
##############################################################################


def get_timestamp():
    """Returns the current timestamp in a specific format."""
    return datetime.now().strftime("%y%m%d-%H%M%S")


def setup_logger(
    logger_name, root, phase, level=logging.INFO, screen=False, tofile=False
):
    """
    Sets up a logger with specified configurations.
    """
    logger = logging.getLogger(logger_name)
    logger.handlers = [] 
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    logger.setLevel(level)

    if tofile:
        log_file = os.path.join(root, f"{phase}_{get_timestamp()}.log")
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger


def dict2str(opt, indent=1):
    """Converts a dictionary to a formatted string for logging."""
    msg = ""
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent * 2) + f"{k}:[\n"
            msg += dict2str(v, indent + 1)
            msg += " " * (indent * 2) + "]\n"
        else:
            msg += " " * (indent * 2) + f"{k}: {v}\n"
    return msg


##############################################################################
# Custom Dataset for Test Images
##############################################################################
class TestImageDataset(Dataset):
    def __init__(self, input_dir, gt_dir, transform=None):
        """
        input_dir: folder path of input images
        gt_dir: folder path of ground-truth images
        """
        self.input_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
        self.gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
        if len(self.input_paths) != len(self.gt_paths):
            raise ValueError(
                f"Mismatch in number of images: {len(self.input_paths)} vs {len(self.gt_paths)}"
            )

        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        # Load input image via PIL and convert to RGB
        input_img = Image.open(self.input_paths[idx]).convert("RGB")
        # Load GT image via PIL
        gt_img = Image.open(self.gt_paths[idx]).convert("RGB")

        # Transform: PIL -> Tensor, range [0,1]
        input_tensor = self.transform(input_img)
        gt_tensor = self.transform(gt_img)

        sample = {
            "input": input_tensor, 
            "gt": gt_tensor,  # ground truth tensor (range [0,1])
            "name": os.path.basename(self.input_paths[idx]),
        }
        return sample


##############################################################################
# Main Test Function with Accelerator and DataLoader
##############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU Test using DataLoader and Accelerator for fast evaluation."
    )
    # Model / Inference arguments
    parser.add_argument(
        "--input_dirs",
        "-i",
        type=str,
        nargs="+",
        required=True,
        help="Folders containing input images for each dataset.",
    )
    parser.add_argument(
        "--gt_dirs",
        "-g",
        type=str,
        nargs="+",
        required=True,
        help="Folders containing GT images for each dataset.",
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        required=True,
        help="Dataset names (one per dataset).",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to the pretrained restoration model.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["fp16", "fp32"],
        default="fp16",
        help="Precision setting for inference.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--process_size",
        type=int,
        default=512,
        help="Minimum size for processing (used for resizing input images).",
    )
    # Logging / Metric arguments
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save log files and results.",
    )
    parser.add_argument(
        "--log_name",
        type=str,
        default="Test_Evaluation",
        help="Base name for log files.",
    )
    parser.add_argument(
        "--test_metric",
        action="store_true",
        help="If True, also logs to a file in log_dir.",
    )
    parser.add_argument(
        "--save_logfile",
        action="store_true",
        help="If True, also logs to a file in log_dir.",
    )
    parser.add_argument(
        "--save_refinement_output",
        action="store_true",
        help="If True, also logs to a file in log_dir.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for test DataLoader."
    )
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    # Initialize Accelerator
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    # Set seeds
    set_seed(args.seed)

    # Setup overall logger (only main process will log final info)
    main_logger = setup_logger(
        "test_main", args.log_dir, args.log_name, screen=True, tofile=args.save_logfile
    )
    main_logger.info("===== Test Configuration =====")
    main_logger.info(dict2str(vars(args)))
    main_logger.info("==============================\n")

    # Initialize IQA metrics (move to device)
    if args.test_metric : 
        iqa_metrics = {
            "PSNR": pyiqa.create_metric("psnr", test_y_channel=False).to(device),
            "SSIM": pyiqa.create_metric("ssim", test_y_channel=False).to(device),
            "LPIPS": pyiqa.create_metric("lpips", device=device),
            "DISTS": pyiqa.create_metric("dists", device=device),
            "MUSIQ": pyiqa.create_metric("musiq", device=device),
            "MANIQA": pyiqa.create_metric("maniqa-pipal", device=device),
            "TOPIQ": pyiqa.create_metric("topiq_nr", device=device),
            "LIQE": pyiqa.create_metric("liqe", device=device),
        }
        fid_metric = pyiqa.create_metric("fid", device=device)

    # For each dataset, create a DataLoader and perform inference & metric computation
    if not (len(args.input_dirs) == len(args.gt_dirs) == len(args.dataset_names)):
        main_logger.error(
            "The number of input_dirs, gt_dirs, and dataset_names must be equal."
        )
        sys.exit(1)

    for dataset_idx, (input_dir, gt_dir, dataset_name) in enumerate(
        zip(args.input_dirs, args.gt_dirs, args.dataset_names)
    ):
        dataset_logger = setup_logger(
            f"test_{dataset_name}",
            args.log_dir,
            args.log_name,
            screen=True,
            tofile=args.save_logfile,
        )
        dataset_logger.info(f"Processing dataset: {dataset_name}")

        # Create Dataset and DataLoader
        test_dataset = TestImageDataset(input_dir, gt_dir, transform=ToTensor())
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        # Prepare with Accelerator (model and dataloader)
        test_loader = accelerator.prepare(test_loader)


        model = ORNet(
            img_channel=4,
            width=64,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 1],
            dec_blk_nums=[1, 1, 1, 1],
        )

        load_net = torch.load(
            args.pretrained_path, map_location=lambda storage, loc: storage
        )
        if "params" in load_net.keys():
            load_net = load_net["params"]
        # remove potential 'module.' prefix if present
        for k, v in deepcopy(load_net).items():
            if k.startswith("module."):
                load_net[k[7:]] = v
                load_net.pop(k)
        model.load_state_dict(load_net, strict=True)
        model.to(device)
        model.eval()

        # Prepare model with Accelerator (for distributed eval)
        model = accelerator.prepare(model)

        cond_list = [0.3]
        for cond in cond_list:
            # Create condition-specific output folder
            cond_folder_name = os.path.join(args.log_dir, dataset_name, f"cond_{cond}")
            os.makedirs(cond_folder_name, exist_ok=True)

            # Setup condition-specific logger
            cond_logger = setup_logger(
                f"test_{dataset_name}_cond_{cond}",
                cond_folder_name,
                args.log_name,
                screen=True,
                tofile=args.save_logfile,
            )
            cond_logger.info(
                f"Processing dataset: {dataset_name} with condition: {cond}"
            )

            # Accumulators for metrics and FID
            if args.test_metric : 
                metrics_accum = {k: 0.0 for k in iqa_metrics.keys()}
                fid_inference_imgs = []  # list to store SR images for FID
                fid_gt_imgs = []  # list to store GT images for FID
            total_images = 0

            t0 = time.time()
            idx = 0
            # Inference loop over DataLoader batches
            with torch.no_grad():
                for batch in tqdm(
                    test_loader,
                    desc=f"Inference ({dataset_name}, cond={cond})",
                    leave=False,
                ):
                    inputs = batch["input"].to(device)  # shape: [B, C, H, W]
                    gts = batch["gt"].to(device)
                    names = batch["name"]

                    # Inference
                    outputs = torch.clamp(
                        model(inputs, cond), 0, 1
                    ) 

                    B = outputs.shape[0]
                    total_images += B


                    # Compute metrics per batch
                    if args.test_metric : 
                        for mname, metric_fn in iqa_metrics.items():
                            if mname in [
                                "MUSIQ",
                                "MANIQA",
                                "TOPIQ",
                                "LIQE",
                                "NIMA"
                            ]:
                                val = metric_fn(outputs).mean().item()
                            else:
                                val = metric_fn(outputs, gts).mean().item()
                            metrics_accum[mname] += val * B

                        # For FID, convert outputs and GT to numpy images (BGR order)
                        for i in range(B):
                            # outputs: tensor in [0,1] -> PIL image -> numpy array
                            out_pil = ToPILImage()(outputs[i].cpu())
                            out_np = np.array(out_pil)  # RGB
                            out_bgr = out_np[..., ::-1]
                            fid_inference_imgs.append(out_bgr)

                            gt_pil = ToPILImage()(gts[i].cpu())
                            gt_np = np.array(gt_pil)
                            gt_bgr = gt_np[..., ::-1]
                            fid_gt_imgs.append(gt_bgr)

                    if args.save_refinement_output : 
                        if idx % 1 == 0:
                            output_save_path = os.path.join(cond_folder_name, names[0])
                            out_pil = ToPILImage()(outputs[0].cpu())
                            # output_save_path = output_save_path.split('.jpg')[0] + '.png'
                            out_pil.save(output_save_path)

                    idx += 1

            t1 = time.time()
            cond_logger.info(
                f"Dataset {dataset_name}, cond {cond}: Processed {total_images} images in {t1 - t0:.2f} seconds."
            )

            
            # Compute average metrics
            if args.test_metric : 
                avg_metrics = {k: (v / total_images) for k, v in metrics_accum.items()}
                metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
                cond_logger.info(
                    f"===== Average Metrics for {dataset_name}, cond {cond} ====="
                )
                cond_logger.info(metric_str)

                # Compute FID using temporary directories
                with tempfile.TemporaryDirectory() as temp_dir_sr, tempfile.TemporaryDirectory() as temp_dir_gt:
                    for i, (sr_img, gt_img) in enumerate(
                        zip(fid_inference_imgs, fid_gt_imgs)
                    ):
                        sr_save_path = os.path.join(temp_dir_sr, f"{i:05d}.jpg")
                        gt_save_path = os.path.join(temp_dir_gt, f"{i:05d}.jpg")
                        cv2.imwrite(sr_save_path, sr_img)
                        cv2.imwrite(gt_save_path, gt_img)
                    fid_start = time.time()
                    fid_value = fid_metric(temp_dir_gt, temp_dir_sr).item()
                    fid_end = time.time()
                cond_logger.info(
                    f"FID: {fid_value:.6f} (computed in {fid_end - fid_start:.2f} sec)"
                )
                cond_logger.info(
                    f"===== Evaluation Completed for dataset: {dataset_name}, cond: {cond} =====\n"
                )


if __name__ == "__main__":
    main()
