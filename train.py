import os
import gc
import lpips
import clip
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from tqdm.auto import tqdm

import diffusers
from diffusers.optimization import get_scheduler
from dataloaders.lqgt_dataset import PairedSROnlineTxtDataset

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs
import wandb
from torchvision.transforms.functional import to_pil_image
import torch.distributed as dist

import random

from arch.ORNet_arch import ORNet
from arch.CFMG_arch import CFMG


# Import pyiqa to compute IQA metrics
from pyiqa import create_metric
from copy import deepcopy

#########################################
#  Validation: Using pyiqa for metrics
#########################################
def validate(model_train, cond, dl_val, args, accelerator, metrics,):

    model_train.eval()
    keys_to_total = list(metrics.keys())

    totals = {key: 0.0 for key in keys_to_total}
    
    count = 0
    with torch.no_grad():
        for batch in dl_val:
            x_src = batch["conditioning_pixel_values"].to(accelerator.device)
            x_tgt = batch["output_pixel_values"].to(accelerator.device)

            # Generate model output: input is [0,1], output is [0,1]
            x_src_input = x_src * 0.5 + 0.5
            x_tgt_pred_0_1 = torch.clamp(model_train(x_src_input, c=cond), 0, 1)

            # For standard IQA metrics, convert to [-1, 1] then back to [0, 1] to match original logic
            x_tgt_pred_m1_1 = x_tgt_pred_0_1 * 2 - 1
            B = x_tgt.size(0)
            count += B

            # Calculate metrics for each image in the batch
            for i in range(B):
                pred_img = (x_tgt_pred_m1_1[i] + 1) / 2.0  # Convert to [0,1] range
                gt_img = (x_tgt[i] + 1) / 2.0
                pred_img_unsqueezed = pred_img.unsqueeze(0)
                gt_img_unsqueezed = gt_img.unsqueeze(0)

                # Standard IQA metrics
                totals["psnr"] += metrics["psnr"](pred_img_unsqueezed, gt_img_unsqueezed).item()
                totals["ssim"] += metrics["ssim"](pred_img_unsqueezed, gt_img_unsqueezed).item()
                totals["lpips"] += metrics["lpips"](pred_img_unsqueezed, gt_img_unsqueezed).item()
                totals["dists"] += metrics["dists"](pred_img_unsqueezed, gt_img_unsqueezed).item()
                totals["musiq"] += metrics["musiq"](pred_img_unsqueezed).item()
                totals["maniqa"] += metrics["maniqa"](pred_img_unsqueezed).item()
                totals["topiq"] += metrics["topiq"](pred_img_unsqueezed).item()
                totals["liqe"] += metrics["liqe"](pred_img_unsqueezed).item()
        
    # Aggregate results across all GPUs
    totals = { key: torch.tensor(val, device=accelerator.device) for key, val in totals.items() }
    count_tensor = torch.tensor(count, device=accelerator.device)

    if dist.is_initialized():
        for key in totals:
            dist.all_reduce(totals[key], op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

    # Calculate average metrics
    avg_metrics = { f"val_{key}_{cond}": (totals[key].item() / count_tensor.item()) for key in totals }

    model_train.train()  # Switch back to training mode
    return avg_metrics


def wandb_init(name, project, config):
    wandb.init(
        id=wandb.util.generate_id(),
        resume="never",
        name=name,
        config=config,
        project=project,
        sync_tensorboard=False,
    )
    print(f"wandb init! name:{name} project:{project}")


def parse_float_list(arg):
    try:
        return [float(x) for x in arg.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be floats")


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
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument( "--revision", type=str,default=None,)
    parser.add_argument("--variant",  type=str,  default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)

    # wandb setting
    parser.add_argument("--wandb_name", default="BeyoundGT")
    parser.add_argument("--wandb_project", default="BeyoundGT")
    parser.add_argument("--wandb_image_log_freq", type=int, default=10)
    parser.add_argument("--log_freq", type=int, default=10)

    # training details
    parser.add_argument("--output_dir", default="experience/BeyoundGT")
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training." )
    parser.add_argument("--resolution",type=int, default=256,)
    parser.add_argument("--train_batch_size", type=int,default=1,help="Batch size (per device) for the training dataloader.",)
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=200000,)
    parser.add_argument("--checkpointing_steps", type=int,default=500,)
    parser.add_argument("--gradient_accumulation_steps",  type=int,default=4,help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing",  action="store_true",)
    parser.add_argument("--train_enhance", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'' "constant", "constant_with_warmup"]' ), )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.",)
    parser.add_argument("--lr_num_cycles", type=int,default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.", )
    parser.add_argument("--lr_power", type=float,default=1.0,help="Power factor of the polynomial scheduler.", )
    parser.add_argument("--dataloader_num_workers", type=int,default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.", )
    parser.add_argument("--adam_beta2",  type=float, default=0.999,help="The beta2 parameter for the Adam optimizer.", )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."  )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer",)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm." )
    parser.add_argument("--allow_tf32", action="store_true",   help="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training.", )
    parser.add_argument("--report_to",type=str, default="tensorboard", help=('Supported platforms are `"tensorboard"`' ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.' ), )
    parser.add_argument("--mixed_precision",type=str, default="fp16",choices=["no", "fp16", "bf16"],  )
    parser.add_argument("--set_grads_to_none", action="store_true",)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--tracker_project_name", type=str,default="train_osediff", help="The name of the wandb project to log to.", )

    parser.add_argument("--train_dataset_txt_paths_list_lq",  type=parse_str_list, default=["YOUR_TRAIN_TXT_FILE_PATH"], help="Comma-separated list of TXT file paths for training", )
    parser.add_argument("--train_dataset_txt_paths_list_gt",   type=parse_str_list,  default=["YOUR_TRAIN_TXT_FILE_PATH"], help="Comma-separated list of TXT file paths for training", )
    parser.add_argument("--train_dataset_txt_paths_list_gt_refined_2x",   type=parse_str_list,  default=["YOUR_TRAIN_TXT_FILE_PATH"], help="Comma-separated list of TXT file paths for training", )
    parser.add_argument("--train_dataset_txt_paths_list_gt_refined_3x",   type=parse_str_list,  default=["YOUR_TRAIN_TXT_FILE_PATH"], help="Comma-separated list of TXT file paths for training", )
    parser.add_argument("--train_dataset_txt_paths_list_gt_refined_4x", type=parse_str_list,  default=["YOUR_TRAIN_TXT_FILE_PATH"],  help="Comma-separated list of TXT file paths for training",)
    parser.add_argument("--test_dataset_txt_paths_list_lq",  type=parse_str_list, default=["YOUR_TEST_TXT_FILE_PATH"], help="Comma-separated list of TXT file paths for testing",)
    parser.add_argument("--test_dataset_txt_paths_list_gt", type=parse_str_list, default=["YOUR_TEST_TXT_FILE_PATH"],help="Comma-separated list of TXT file paths for testing",)
    parser.add_argument("--dataset_prob_paths_list", type=parse_int_list,default=[1], help="A comma-separated list of integers",)
    parser.add_argument("--lambda_l2", default=1.0, type=float)


    parser.add_argument("--validation_steps", type=int, default=10000,  help="Run validation every N training steps.", )
    parser.add_argument("--pretrained_CFMG_path", type=str, default=None, help="Pretrained OSEDDiff" )


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):

    if args.seed is not None:
        set_seed(args.seed)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

        print("seed : ", args.seed)

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

    if accelerator.is_main_process:
        wandb_init(args.wandb_name, args.wandb_project, dict(vars(args)))

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)



    if args.train_enhance:
        model_train = CFMG(
            img_channel=7,
            output_channel=4,
            width=32,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 1],
            dec_blk_nums=[1, 1, 1, 1],
        )
    else:
        model_fixed = CFMG(
            img_channel=7,
            output_channel=4,
            width=32,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 1],
            dec_blk_nums=[1, 1, 1, 1],
        )

        load_net = torch.load(
            args.pretrained_CFMG_path, map_location=lambda storage, loc: storage
        )
        if "params" in load_net.keys():
            load_net = load_net["params"]
        # remove potential 'module.' prefix if present
        for k, v in deepcopy(load_net).items():
            if k.startswith("module."):
                load_net[k[7:]] = v
                load_net.pop(k)
        model_fixed.load_state_dict(load_net, strict=True)
        model_fixed.to(device)
        model_fixed.eval()

    
        model_train = ORNet(
            img_channel=4,
            width=64,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 1],
            dec_blk_nums=[1, 1, 1, 1],
        )
        model_train = model_train.train()

    model_train_layers_to_opt = []
    for n, _p in model_train.named_parameters():
        model_train_layers_to_opt.append(_p)

    model_optimizer = torch.optim.AdamW(
        model_train_layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    model_lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=model_optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    if args.train_enhance:

        net_musiq = create_metric("musiq", as_loss=True).cuda()
        net_musiq.requires_grad_(False)

        net_maniqa = create_metric("maniqa-pipal", as_loss=True).cuda()
        net_maniqa.requires_grad_(False)

        net_topiq = create_metric("topiq_nr", as_loss=True).cuda()
        net_topiq.requires_grad_(False)

        net_topiq = accelerator.prepare(net_topiq)
        net_musiq = accelerator.prepare(net_musiq)
        net_maniqa = accelerator.prepare(net_maniqa)

    else : 
        # net_lpips = lpips.LPIPS(net="vgg").cuda()
        net_lpips = create_metric("lpips", as_loss=True).cuda()
        net_lpips.requires_grad_(False)
        net_lpips = accelerator.prepare(net_lpips)

        net_dists = create_metric("dists", as_loss=True).cuda()
        net_dists.requires_grad_(False)
        net_dists = accelerator.prepare(net_dists)

    dataset_train = PairedSROnlineTxtDataset(split="train", args=args)
    dataset_val = PairedSROnlineTxtDataset(split="test", args=args)
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        worker_init_fn=worker_init_fn,
    )
    dl_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=0
    )

    if args.train_enhance:
        dl_train, dl_val, model_train, model_optimizer, model_lr_scheduler = (
            accelerator.prepare(  dl_train, dl_val, model_train, model_optimizer, model_lr_scheduler ) )
    else:
        dl_train, dl_val, model_train,model_optimizer,model_lr_scheduler, model_fixed = accelerator.prepare(
            dl_train, dl_val, model_train,  model_optimizer,  model_lr_scheduler, model_fixed )


    # renorm with image net statistics
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    if accelerator.is_main_process:
        args.train_dataset_txt_paths_list_lq = str(args.train_dataset_txt_paths_list_lq)
        args.train_dataset_txt_paths_list_gt = str(args.train_dataset_txt_paths_list_gt)
        args.train_dataset_txt_paths_list_gt_refined_2x = str(
            args.train_dataset_txt_paths_list_gt_refined_2x
        )
        args.train_dataset_txt_paths_list_gt_refined_3x = str(
            args.train_dataset_txt_paths_list_gt_refined_3x
        )
        args.train_dataset_txt_paths_list_gt_refined_4x = str(
            args.train_dataset_txt_paths_list_gt_refined_4x
        )
        args.test_dataset_txt_paths_list_lq = str(args.test_dataset_txt_paths_list_lq)
        args.test_dataset_txt_paths_list_gt = str(args.test_dataset_txt_paths_list_gt)
        args.dataset_prob_paths_list = str(args.dataset_prob_paths_list)
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    metrics = {
        "psnr": create_metric("psnr"),
        "ssim": create_metric("ssim"),
        "lpips": create_metric("lpips"),
        "dists": create_metric("dists"),
        "musiq": create_metric("musiq"),
        "maniqa": create_metric("maniqa-pipal"),
        "liqe": create_metric("liqe"),
        "topiq": create_metric("topiq_nr"),
    }

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # start the training loop
    global_step = 0

    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            if args.train_enhance:
                m_acc = [model_train]
            else:
                m_acc = [model_train, model_fixed]

            
            # END OF MODIFIED SECTION

            with accelerator.accumulate(*m_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                x_tgt_refined_2x = batch["output_refiend_pixel_values_2x"]
                x_tgt_refined_3x = batch["output_refiend_pixel_values_3x"]
                x_tgt_refined_4x = batch["output_refiend_pixel_values_4x"]
                B, C, H, W = x_src.shape
                # # # get text prompts from GT

                B = x_src.size(0)
                cond = torch.rand(1).item()
    
                x_src_input = x_tgt * 0.5 + 0.5
                x_original_lq = x_src * 0.5 + 0.5

                if args.train_enhance:
                    mix_weights, value_prob = model_train(x_src_input, c=cond)

                    imgs = [
                        x_tgt,  
                        x_tgt_refined_2x,  # 2x refined
                        x_tgt_refined_3x,  # 3x refined
                        x_tgt_refined_4x,  # 4x refined
                    ]
                    # ****************************** frequency mixup  ******************************
                    freqs = []
                    for img in imgs:
                        F_img = torch.fft.fftn(img, dim=(-2, -1), norm="ortho")
                        F_img = torch.fft.fftshift(F_img, dim=(-2, -1))
                        freqs.append(F_img)

                    weighted_freqs = []
                    for i, F_img in enumerate(freqs):
                        w = mix_weights[:, i].unsqueeze(1)
                        weighted_freqs.append(F_img * w)

                    freq_mix = sum(weighted_freqs)  # [B, C, H, W], complex
                    freq_mix = torch.fft.ifftshift(freq_mix, dim=(-2, -1))
                    x_tgt_pred = torch.fft.ifftn(
                        freq_mix, dim=(-2, -1), norm="ortho"
                    ).real

                    loss_mse = (
                        F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean")
                        * args.lambda_l2
                    )

                    loss_topiq = net_topiq(x_tgt_pred.float()).mean()
                    loss_musiq = net_musiq(x_tgt_pred.float()).mean()
                    loss_maniqa = net_maniqa(x_tgt_pred.float()).mean()

                    perceptual_loss = (
                        loss_topiq + loss_musiq / 100 + loss_maniqa
                    ) * 0.1

                    loss = loss_mse * (1 - cond * cond) - perceptual_loss * cond * cond
                    total_loss = loss
                else:
                    with torch.no_grad():
                        mix_weights, value_prob = model_fixed(x_src_input, c=cond)

                        imgs = [
                            x_tgt, 
                            x_tgt_refined_2x,  # 2x refined
                            x_tgt_refined_3x,  # 3x refined
                            x_tgt_refined_4x,  # 4x refined
                        ]

                        freqs = []
                        for img in imgs:
                            F_img = torch.fft.fftn(img, dim=(-2, -1), norm="ortho")
                            F_img = torch.fft.fftshift(F_img, dim=(-2, -1))
                            freqs.append(F_img)

                        weighted_freqs = []
                        for i, F_img in enumerate(freqs):
                            w = mix_weights[:, i].unsqueeze(1)
                            weighted_freqs.append(F_img * w)

                        freq_mix = sum(weighted_freqs)  # [B, C, H, W], complex
                        freq_mix = torch.fft.ifftshift(freq_mix, dim=(-2, -1))
                        x_tgt_mixed = torch.clamp( torch.fft.ifftn(freq_mix, dim=(-2, -1), norm="ortho").real, -1, 1, )

                    x_tgt_pred = torch.clamp(model_train(x_src_input, c=cond), 0, 1)
                    x_tgt_pred = x_tgt_pred * 2 - 1

                    loss = F.l1_loss(
                        x_tgt_pred.float(), x_tgt_mixed.float(), reduction="mean"
                    )
                    loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean()
                    loss_dists = net_dists(x_tgt_pred.float(), x_tgt.float()).mean()
                    fidelity_loss = (loss_lpips + loss_dists)*0.03
                    total_loss = loss + fidelity_loss
      

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model_train_layers_to_opt, args.max_grad_norm
                    )
                model_optimizer.step()
                model_lr_scheduler.step()
                model_optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
     
                    logs["loss"] = loss.detach().item()
                    progress_bar.set_postfix(**logs)

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 0:
                        outf = os.path.join( args.output_dir, "checkpoints",  f"model_train_{global_step}.pkl", )
                        accelerator.unwrap_model(model_train).save_model(outf)

                    # accelerator.log(logs, step=global_step)
                    if global_step % args.log_freq == 0:
                        print(f"step : {global_step}", logs)

                    if global_step % args.log_freq == 0:
                        wandb.log(logs, step=global_step)

            if not args.train_enhance:
                val_cond_list = [0.3]

                if global_step % args.validation_steps == 0:
                    for val_cond in val_cond_list:

                        val_metrics = validate(
                            model_train, val_cond, dl_val, args, accelerator, metrics
                        )

                        if accelerator.is_main_process:
                            wandb.log(val_metrics, step=global_step)
                            print(  f"Validation metrics at step {global_step}: {val_metrics}" )

            if accelerator.is_main_process:
                if global_step % args.wandb_image_log_freq == 0:
                    image_input_array = []
                    image_output_array = []
                    image_gt_array = []
                    image_gt_refined_array_2x = []
                    image_gt_refined_array_3x = []
                    image_gt_refined_array_4x = []
                    image_update_array = []
                    image_fix_array = []
                    image_cond_array = []
                    image_mixup_rate_array = []
                    image_mixed_array = []

                    for i in range(x_src.size(0)):
                        image_input_array.append(to_pil_image(torch.clamp((x_src[i] + 1) / 2, 0, 1)))
                        image_output_array.append(to_pil_image(torch.clamp((x_tgt_pred[i] + 1) / 2, 0, 1)))
                        image_gt_array.append(to_pil_image(torch.clamp((x_tgt[i] + 1) / 2, 0, 1)))
                        image_gt_refined_array_2x.append(to_pil_image(torch.clamp((x_tgt_refined_2x[i] + 1) / 2, 0, 1)))
                        image_gt_refined_array_3x.append(to_pil_image(torch.clamp((x_tgt_refined_3x[i] + 1) / 2, 0, 1)))
                        image_gt_refined_array_4x.append(to_pil_image(torch.clamp((x_tgt_refined_4x[i] + 1) / 2, 0, 1)))

                        for j in range(mix_weights.size(1)):
                            image_mixup_rate_array.append( to_pil_image(torch.clamp((mix_weights[:, j].unsqueeze(1)[i]), 0, 1 ) ))
                        if not args.train_enhance:
                            image_mixed_array.append( to_pil_image( torch.clamp((x_tgt_mixed[i] + 1) / 2, 0, 1)) )

                    wandb.log({"images/Input": [ wandb.Image(image) for image in image_input_array ] }, step=global_step,)
                    wandb.log({"images/Output": [ wandb.Image(image, caption=f"cond: {cond}")for image in image_output_array  ] }, step=global_step, )
                    wandb.log({"images/gt": [ wandb.Image(image) for image in image_gt_array]}, step=global_step,)
                    wandb.log({"images/gt_refined_2x": [ wandb.Image(image)for image in image_gt_refined_array_2x] },step=global_step,)
                    wandb.log({"images/gt_refined_3x": [ wandb.Image(image) for image in image_gt_refined_array_3x ]  },step=global_step, )
                    wandb.log({"images/gt_refined_4x": [ wandb.Image(image) for image in image_gt_refined_array_4x ]  },step=global_step, )
                    wandb.log({"images/mixup_weight": [ wandb.Image(image) for image in image_mixup_rate_array ]   },step=global_step, )
                    if not args.train_enhance:
                        wandb.log({"images/mixed": [ wandb.Image(image) for image in image_mixed_array  ]}, step=global_step, )


            if global_step > args.max_train_steps:
                exit()


if __name__ == "__main__":
    args = parse_args()
    main(args)