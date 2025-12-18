import os
import argparse
import numpy as np
import torch
import transformers
import diffusers
import random

from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs
from tqdm.auto import tqdm
from pathlib import Path
from copy import deepcopy
from torchvision.transforms.functional import to_pil_image


from dataloaders.lqgt_dataset_whole import PairedSROnlineTxtDataset
from arch.CFMG_arch import CFMG

def save_cond_variations(
    model,
    x_src,
    x_tgt,
    x_tgt_refined_2x,
    x_tgt_refined_3x,
    x_tgt_refined_4x,
    output_dir,
    device,
    name,
    num_channels=4,
):
    os.makedirs(output_dir, exist_ok=True)
    # normalize input and target to [0,1]
    x_src_input = (x_tgt * 0.5 + 0.5).to(device)
    B = x_src_input.size(0)

    # iterate cond values
    cond_val_list = [0.3]
    for cond_val in cond_val_list:
        cond_str = f"{cond_val:.1f}"
        subdir = os.path.join(output_dir, f"cond_{cond_str}")
        os.makedirs(subdir, exist_ok=True)

        # forward pass
        with torch.no_grad():
            mix_weights, _ = model(x_src_input, c=cond_val)
            
            # FFT mixing logic
            imgs = [
                x_tgt,
                x_tgt_refined_2x,
                x_tgt_refined_3x,
                x_tgt_refined_4x,
            ] 
            freqs = [
                torch.fft.fftshift(
                    torch.fft.fftn(img.to(device), norm="ortho"), dim=(-2, -1)
                )
                for img in imgs
            ]
            weighted = [
                freqs[i] * mix_weights[:, i].unsqueeze(1) for i in range(len(freqs))
            ]
            freq_mix = sum(weighted)
            freq_mix = torch.fft.ifftn(
                torch.fft.ifftshift(freq_mix, dim=(-2, -1)), norm="ortho"
            ).real
            x_tgt_pred = freq_mix

        # save per-sample
        for i in range(B):
            mixed = to_pil_image((x_tgt_pred[i].clamp(-1, 1) * 0.5 + 0.5).cpu())

            bname_with_ext = os.path.basename(name)
            bname_without_ext, _ = os.path.splitext(bname_with_ext)

            output_filename = f"{bname_without_ext}.png"
            mixed.save(os.path.join(subdir, output_filename))


# --- Argument Parsing Helper Functions ---
def parse_int_list(arg):
    try:
        return [int(x) for x in arg.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be integers")


def parse_str_list(arg):
    return arg.split(",")


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()

    # Basic Configuration
    parser.add_argument("--output_dir", default="experience")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    
    # Model & Data Configuration
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    

    # Accelerator Settings
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='Supported platforms are "tensorboard", "wandb", "comet_ml", or "all".',
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
    )
    # Gradient accumulation is needed for accelerator init, though unused in inference
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4) 
    parser.add_argument("--tracker_project_name", type=str, default="train")

    # Dataset Paths (Only Train split kept as code uses split='train')
    parser.add_argument("--train_dataset_txt_paths_list_lq", type=parse_str_list, default=["YOUR_PATH"])
    parser.add_argument("--train_dataset_txt_paths_list_gt", type=parse_str_list, default=["YOUR_PATH"])
    parser.add_argument("--train_dataset_txt_paths_list_gt_refined_2x", type=parse_str_list, default=["YOUR_PATH"])
    parser.add_argument("--train_dataset_txt_paths_list_gt_refined_3x", type=parse_str_list, default=["YOUR_PATH"])
    parser.add_argument("--train_dataset_txt_paths_list_gt_refined_4x", type=parse_str_list, default=["YOUR_PATH"])
    
    parser.add_argument("--dataset_prob_paths_list", type=parse_int_list, default=[1])
    
    # Dataset specific param (kept in case dataset class needs it)
    parser.add_argument("--deg_file_path", default="params_realesrgan.yml", type=str)

    # Model Weights Paths
    parser.add_argument("--pretrained_CFMG_path", type=str, default=None, help="Path to Pretrained Model")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    # 1. Seed Setting
    if args.seed is not None:
        set_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        print("seed : ", args.seed)

    # 2. Accelerator Setup
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device

    # Logging verbosity
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # 3. Model Setup
    model = CFMG(
            img_channel=7,
            output_channel=4,
            width=32,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 1],
            dec_blk_nums=[1, 1, 1, 1],
        )
    
    # Load Pretrained Weights
    load_net = torch.load(
        args.pretrained_CFMG_path, map_location=lambda storage, loc: storage
    )
    if "params" in load_net.keys():
        load_net = load_net["params"]
    
    # Remove 'module.' prefix if present
    for k, v in deepcopy(load_net).items():
        if k.startswith("module."):
            load_net[k[7:]] = v
            load_net.pop(k)
            
    model.load_state_dict(load_net, strict=True)
    model.to(device)
    model.eval() # Inference Mode

    # 4. Dataset & Dataloader
    # Note: Code uses 'train' split for loading data
    dataset_train = PairedSROnlineTxtDataset(split="train", args=args)
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True, # Shuffle kept true as in original, set to False if sequential processing is needed
        num_workers=args.dataloader_num_workers,
        worker_init_fn=worker_init_fn,
    )

    dl_train, model = accelerator.prepare(dl_train, model)

    # 5. Tracker Init
    if accelerator.is_main_process:
        # Convert list args to string for logging
        args.train_dataset_txt_paths_list_lq = str(args.train_dataset_txt_paths_list_lq)
        args.train_dataset_txt_paths_list_gt = str(args.train_dataset_txt_paths_list_gt)
        args.train_dataset_txt_paths_list_gt_refined_2x = str(args.train_dataset_txt_paths_list_gt_refined_2x)
        args.train_dataset_txt_paths_list_gt_refined_3x = str(args.train_dataset_txt_paths_list_gt_refined_3x)
        args.train_dataset_txt_paths_list_gt_refined_4x = str(args.train_dataset_txt_paths_list_gt_refined_4x)
        args.dataset_prob_paths_list = str(args.dataset_prob_paths_list)
        
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # 6. Main Loop (Inference)
    progress_bar = tqdm(
        range(0, len(dl_train)),
        initial=0,
        desc="Processing",
        disable=not accelerator.is_local_main_process,
    )

    for step, batch in enumerate(dl_train):

        # Inference only, removed accelerator.accumulate()
        x_src = batch["conditioning_pixel_values"]
        x_tgt = batch["output_pixel_values"]
        x_tgt_refined_2x = batch["output_refiend_pixel_values_2x"]
        x_tgt_refined_3x = batch["output_refiend_pixel_values_3x"]
        x_tgt_refined_4x = batch["output_refiend_pixel_values_4x"]
        
        name_list = batch["name"]
        name = name_list[0].split("/")[-1] 

        save_cond_variations(
            model,
            x_src,
            x_tgt,
            x_tgt_refined_2x,
            x_tgt_refined_3x,
            x_tgt_refined_4x,
            args.output_dir,
            accelerator.device,
            name,
            num_channels=4,
        )
        
        progress_bar.update(1)

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)