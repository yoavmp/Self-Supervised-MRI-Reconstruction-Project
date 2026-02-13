# System / Python
import os
import argparse
import logging
import random
import shutil
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
# Custom
from net.network import ParallelNetwork, SingleNetwork
from fastmri_dataset import FastMRIData as Dataset
from mri_tools import rA, rAtA, rfft2
from utils import psnr_slice, ssim_slice

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', type=str, default='self-supervised MRI reconstruction', help='name of experiment')
# parameters related to distributed training
parser.add_argument('--init-method', default='tcp://localhost:1836', help='initialization method')
parser.add_argument('--nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--gpus', type=int, default=torch.cuda.device_count(), help='number of gpus per node')
parser.add_argument('--world-size', type=int, default=None, help='world_size = nodes * gpus')
# parameters related to model
parser.add_argument('--use-init-weights', '-uit', type=bool, default=True, help='whether initialize model weights with defined types')
parser.add_argument('--init-type', type=str, default='xavier', help='type of initialize model weights')
parser.add_argument('--gain', type=float, default=1.0, help='gain in the initialization of model weights')
parser.add_argument('--num-layers', type=int, default=9, help='number of iterations')
# learning rate, batch size, and etc
parser.add_argument('--seed', type=int, default=30, help='random seed number')
parser.add_argument('--lr', '-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--batch-size', type=int, default=4, help='batch size of single gpu')
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
parser.add_argument('--warmup-epochs', type=int, default=10, help='number of warmup epochs')
parser.add_argument('--num-epochs', type=int, default=500, help='maximum number of epochs')
# parameters related to data and masks
parser.add_argument('--train-path', type=str, default='/home/hc/data/IXI_T1/train', help='path of training data')
parser.add_argument('--val-path', type=str, default='/home/hc/data/IXI_T1/val', help='path of validation data')
parser.add_argument('--test-path', type=str, default='/home/hc/data/IXI_T1/test', help='path of test data')

parser.add_argument('--train-file-list', type=str, default=None, help='optional text file of .h5 volume paths for training split (one path per line)')
parser.add_argument('--val-file-list', type=str, default=None, help='optional text file of .h5 volume paths for validation split (one path per line)')
parser.add_argument('--test-file-list', type=str, default=None, help='optional text file of .h5 volume paths for test split (one path per line)')
parser.add_argument('--u-mask-path', type=str, default='./mask/undersampling_mask/mask_8.00x_acs24.mat', help='undersampling mask')
parser.add_argument('--s-mask-up-path', type=str, default='./mask/selecting_mask/mask_2.00x_acs16.mat', help='selection mask in up network')
parser.add_argument('--s-mask-down-path', type=str, default='./mask/selecting_mask/mask_2.50x_acs16.mat', help='selection mask in down network')
parser.add_argument('--train-sample-rate', '-trsr', type=float, default=0.06, help='sampling rate of training data')
parser.add_argument('--val-sample-rate', '-vsr', type=float, default=0.02, help='sampling rate of validation data')
parser.add_argument('--test-sample-rate', '-tesr', type=float, default=0.02, help='sampling rate of test data')
# save path
parser.add_argument('--model-save-path', type=str, default='./checkpoints/', help='save path of trained model')
parser.add_argument('--loss-curve-path', type=str, default='./runs/loss_curve/', help='save path of loss curve in tensorboard')
# others
parser.add_argument('--mode', '-m', type=str, default='train', help='whether training or test model, value should be set to train or test')
parser.add_argument('--pretrained', '-pt', action='store_true', help='whether load checkpoint')
# SSDU comparison
parser.add_argument('--ssdu', action='store_true', help='Train SSDU baseline (single net, theta/lambda split)')
parser.add_argument('--ssdu-theta-frac', type=float, default=0.5, help='Fraction of acquired samples assigned to theta')
# Save imgaes
parser.add_argument('--save-samples', action='store_true', help='Save a few recon samples during test')
parser.add_argument('--num-save-samples', type=int, default=5)
parser.add_argument('--samples-dir', type=str, default=None, help='Where to save pngs (defaults under run dir)')
parser.add_argument('--reference-files', type=str, default=None,
                    help='Path to txt file: one full .h5 path per line')
parser.add_argument('--reference-pairs', type=str, default=None,
                    help='Optional: path to txt file with lines: <fullpath>\t<slice_id>')
# supervised ista comparison
parser.add_argument('--supervised-ista', action='store_true',
                    help='Supervised ISTA-Net+ baseline (image MSE vs GT).')


def create_logger():
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s:\t%(message)s')
    stream_formatter = logging.Formatter('%(levelname)s:\t%(message)s')

    file_handler = logging.FileHandler(filename='logger.txt', mode='a+', encoding='utf-8')
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def init_weights(net, init_type='xavier', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('Initialization method {} is not implemented.'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


class EarlyStopping:
    def __init__(self, patience=50, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, metrics, loss=True):
        score = -metrics if loss else metrics
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def forward(mode, rank, model, dataloader, criterion, optimizer, log, args):
    assert mode in ['train', 'val', 'test']
    loss, psnr, ssim = 0.0, 0.0, 0.0
    metric_n = 0  # count only slices that actually have labels (fastMRI test split may not)

    saved_n = 0
    want_save = (mode == "test") and getattr(args, "save_samples", False) and (rank == 0)


    # ----------------------------
    # Save a few sample PNGs in test
    # ----------------------------
    want_save = (mode == "test") and getattr(args, "save_samples", False) and (rank == 0)
    saved_n = 0


    def _load_reference_files(path):
        files = []
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                files.append(s)
        return files

    def _load_reference_pairs(path):
        pairs = set()
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                fp, sid = s.split("\t")
                pairs.add((fp, int(sid)))
        return pairs

    ref_files = None
    ref_pairs = None

    if want_save:
        if getattr(args, "samples_dir", None) is None:
            args.samples_dir = os.path.join(os.path.dirname(args.model_save_path), "samples")
        os.makedirs(args.samples_dir, exist_ok=True)

        # Load reference file list (required for auto-pick)
        if getattr(args, "reference_files", None):
            ref_files = set(_load_reference_files(args.reference_files))
        else:
            ref_files = None  # means "no filtering"

        # If provided, use exact slice pairs (for perfect matching across runs)
        if getattr(args, "reference_pairs", None):
            ref_pairs = _load_reference_pairs(args.reference_pairs)
        else:
            ref_pairs = None

        # store picked pairs during this run (first model run)
        if not hasattr(args, "_picked_pairs"):
            args._picked_pairs = {}  # dict: file_path -> slice_id

    t = tqdm(dataloader, desc=mode + 'ing', total=int(len(dataloader))) if rank == 0 else dataloader
    for iter_num, data_batch in enumerate(t):
        # Dataset returns: label, mask_under, mask_net_up, mask_net_down, fname, slice_id, has_label
        label, mask_under, mask_net_up, mask_net_down, fname, slice_id, has_label = data_batch

        label = label.to(rank, non_blocking=True)
        mask_under = mask_under.to(rank, non_blocking=True)
        mask_net_up = mask_net_up.to(rank, non_blocking=True)
        mask_net_down = mask_net_down.to(rank, non_blocking=True)

        # has_label can come as bool, list[bool], or tensor depending on collate/batch
        if torch.is_tensor(has_label):
            has_label_list = has_label.detach().cpu().bool().tolist()
        elif isinstance(has_label, (list, tuple)):
            has_label_list = [bool(x) for x in has_label]
        else:
            has_label_list = [bool(has_label)]

        under_img = rAtA(label, mask_under)
        under_kspace = rA(label, mask_under)

        # ----------------------------
        # Supervised ISTA-Net+ branch (SingleNetwork, supervised loss)
        # ----------------------------
        if getattr(args, "supervised_ista", False):
            # Inference input: zero-filled / normal equation backproj (same as your code)
            inp_img  = under_img
            inp_mask = mask_under

            output, _ = model(inp_img.permute(0, 3, 1, 2).contiguous(), inp_mask)  # NCHW

            if rank == 0 and not hasattr(args, "_printed_ista_return"):
                args._printed_ista_return = True
                out0, out1 = model(inp_img.permute(0,3,1,2).contiguous(), inp_mask)
                print("[SANITY] model returns types/shapes:",
                    type(out0), getattr(out0, "shape", None),
                    type(out1), getattr(out1, "shape", None))

            output = output.permute(0, 2, 3, 1).contiguous()                       # NHWC

            # -------- optional: SAVE SAMPLE PNGs (test only) --------
            if want_save and saved_n < args.num_save_samples:
                B = label.shape[0]
                for i in range(B):
                    if saved_n >= args.num_save_samples:
                        break
                    if not has_label_list[i]:
                        continue

                    f_full = str(fname[i] if isinstance(fname, (list, tuple)) else fname)
                    sid = int(slice_id[i] if isinstance(slice_id, (list, tuple)) else slice_id)

                    if iter_num == 0 and rank == 0:
                        print("SAVING PNGs TO:", args.samples_dir)
                        print("DEBUG fname example:", f_full)
                        if ref_files is not None:
                            ex = next(iter(ref_files))
                            print("DEBUG ref_files example:", ex)
                            print("DEBUG basename match:", os.path.basename(f_full) == os.path.basename(ex))

                    if ref_files is not None and f_full not in ref_files:
                        continue

                    if ref_pairs is not None:
                        if (f_full, sid) not in ref_pairs:
                            continue
                    else:
                        if f_full not in args._picked_pairs:
                            args._picked_pairs[f_full] = sid
                        if args._picked_pairs[f_full] != sid:
                            continue

                    # Use a consistent display scale (0..1) for MRI magnitude-like images
                    def clamp01(x):
                        return np.clip(x.astype(np.float32), 0.0, 1.0)

                    recon = output[i].detach().float().cpu().numpy()[:, :, 0]
                    gt    = label[i].detach().float().cpu().numpy()[:, :, 0]
                    diff  = np.abs(recon - gt)

                    recon_vis = clamp01(recon)
                    gt_vis    = clamp01(gt)
                    diff_vis = clamp01(diff)

                    base = os.path.join(
                        args.samples_dir,
                        f"sample{saved_n:02d}_{os.path.basename(f_full)}_slice{sid:03d}"
                    )

                    plt.imsave(base + "_recon.png",   recon_vis, cmap="gray", vmin=0.0, vmax=1.0)
                    plt.imsave(base + "_gt.png",      gt_vis,    cmap="gray", vmin=0.0, vmax=1.0)
                    plt.imsave(base + "_absdiff.png", diff_vis,  cmap="gray")  # (or vmin/vmax if using clamp01)


                    saved_n += 1
            # -------------------------------------------------------

            # Supervised image-domain loss and metrics (only where GT exists)
            B = label.shape[0]
            valid_n = 0
            batch_loss = 0.0

            for i in range(B):
                if (mode == 'val') or has_label_list[i]:
                    batch_loss = batch_loss + criterion(output[i:i+1], label[i:i+1])
                    valid_n += 1

                    if mode in ['val', 'test']:
                        psnr += psnr_slice(label[i:i+1], output[i:i+1])
                        ssim += ssim_slice(label[i:i+1], output[i:i+1])
                        metric_n += 1

            # avoid div-by-zero if a batch contains no labeled slices
            if valid_n > 0:
                batch_loss = batch_loss / valid_n
            else:
                batch_loss = torch.zeros((), device=label.device)

            if mode == 'train':
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            loss += float(batch_loss.item())
            continue  # skip SSDU + parallel branches


        # ----------------------------
        # SSDU branch (single network)
        # ----------------------------
        if args.ssdu and mode == 'train':
            mask_theta  = mask_net_up    # Θ: goes into network input
            mask_lambda = mask_net_down  # Λ: held-out acquired points used only for loss

            theta_img = rAtA(label, mask_theta)

            output, loss_layers = model(theta_img.permute(0,3,1,2).contiguous(), mask_theta)  # output is NCHW
            output = output.permute(0,2,3,1).contiguous()                                     # now NHWC (…,1)
            pred_lambda = rA(output, mask_lambda)
            tgt_lambda  = under_kspace * mask_lambda
            ssdu_loss = criterion(pred_lambda, tgt_lambda)

            # same constraint regularizer pattern you already have
            constr_loss = criterion(loss_layers[0], torch.zeros_like(loss_layers[0]))
            for i in range(args.num_layers - 1):
                constr_loss += criterion(loss_layers[i + 1], torch.zeros_like(loss_layers[i + 1]))

            batch_loss = ssdu_loss + 0.01 * constr_loss

            if mode == 'train':
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            loss += batch_loss.item()
            continue  # skip the rest of the parallel-network code for this batch

        #######################
        # ----------------------------
        # SSDU eval branch (single network)
        # ----------------------------
        if args.ssdu and mode in ['val', 'test']:
            # For evaluation, feed the full acquired mask (mask_under) just like standard inference
            eval_img = under_img
            eval_mask = mask_under

            output, _ = model(eval_img.permute(0,3,1,2).contiguous(), eval_mask)
            output = output.permute(0,2,3,1).contiguous()

            # ----------------------------
            # SAVE SAMPLE PNGs (test only)
            # ----------------------------
            if want_save and saved_n < args.num_save_samples:
                B = label.shape[0]
                for i in range(B):
                    if saved_n >= args.num_save_samples:
                        break
                    if not has_label_list[i]:
                        continue

                    f_full = str(fname[i] if isinstance(fname, (list, tuple)) else fname)
                    sid = int(slice_id[i] if isinstance(slice_id, (list, tuple)) else slice_id)

                    if iter_num == 0 and rank == 0:
                        print("SAVING PNGs TO:", args.samples_dir)
                        print("DEBUG fname example:", f_full)
                        if ref_files is not None:
                            ex = next(iter(ref_files))
                            print("DEBUG ref_files example:", ex)
                            print("DEBUG basename match:", os.path.basename(f_full) == os.path.basename(ex))

                    # optional filter: only consider selected reference files
                    if ref_files is not None and f_full not in ref_files:
                        continue

                    # exact-match mode: save only listed (file, slice) pairs
                    if ref_pairs is not None:
                        if (f_full, sid) not in ref_pairs:
                            continue
                    else:
                        # auto-pick mode: pick exactly 1 slice per file and stick to it
                        if f_full not in args._picked_pairs:
                            args._picked_pairs[f_full] = sid
                        if args._picked_pairs[f_full] != sid:
                            continue

                    recon = output[i:i+1].detach().float().cpu().numpy()[0, :, :, 0]
                    gt    = label[i:i+1].detach().float().cpu().numpy()[0, :, :, 0]
                    diff  = np.abs(recon - gt)

                    def norm01(x):
                        x = x.astype(np.float32)
                        x = x - x.min()
                        return x / (x.max() + 1e-8)

                    base = os.path.join(
                        args.samples_dir,
                        f"sample{saved_n:02d}_{os.path.basename(f_full)}_slice{sid:03d}"
                    )
                    plt.imsave(base + "_recon.png",   norm01(recon), cmap="gray")
                    plt.imsave(base + "_gt.png",      norm01(gt),    cmap="gray")
                    plt.imsave(base + "_absdiff.png", norm01(diff),  cmap="gray")

                    saved_n += 1
            # ----------------------------

            # optional: define a consistent loss for logging (on acquired points)
            pred_under = rA(output, mask_under)
            batch_loss = criterion(pred_under, under_kspace)


            # metrics only if label exists (val always should)
            B = label.shape[0]
            for i in range(B):
                if (mode == 'val') or has_label_list[i]:
                    psnr += psnr_slice(label[i:i+1], output[i:i+1])
                    ssim += ssim_slice(label[i:i+1], output[i:i+1])
                    metric_n += 1

            loss += batch_loss.item()
            continue  # skip parallel-network path

        net_img_up = rAtA(label, mask_net_up)
        net_img_down = rAtA(label, mask_net_down)

        if mode == 'test':
            net_img_up = net_img_down = under_img
            mask_net_up = mask_net_down = mask_under

        output_up, loss_layers_up, output_down, loss_layers_down = model(
            net_img_up.permute(0, 3, 1, 2).contiguous(), mask_net_up,
            net_img_down.permute(0, 3, 1, 2).contiguous(), mask_net_down
        )
        output_up = output_up.permute(0, 2, 3, 1).contiguous()
        output_down = output_down.permute(0, 2, 3, 1).contiguous()

        # ----------------------------
        # SAVE SAMPLE PNGs (test only)
        # ----------------------------
        if want_save and saved_n < args.num_save_samples:
            B = label.shape[0]

            for i in range(B):
                if saved_n >= args.num_save_samples:
                    break
                if not has_label_list[i]:
                    continue

                f_full = str(fname[i] if isinstance(fname, (list, tuple)) else fname)
                sid = int(slice_id[i] if isinstance(slice_id, (list, tuple)) else slice_id)

                if iter_num == 0 and rank == 0:
                    print("SAVING PNGs TO:", args.samples_dir)
                    print("DEBUG fname example:", f_full)
                    if ref_files is not None:
                        ex = next(iter(ref_files))
                        print("DEBUG ref_files example:", ex)
                        print("DEBUG basename match:", os.path.basename(f_full) == os.path.basename(ex))

                if ref_files is not None and f_full not in ref_files:
                    continue

                if ref_pairs is not None:
                    if (f_full, sid) not in ref_pairs:
                        continue
                else:
                    if f_full not in args._picked_pairs:
                        args._picked_pairs[f_full] = sid
                    if args._picked_pairs[f_full] != sid:
                        continue

                recon = output_up[i:i+1].detach().float().cpu().numpy()[0, :, :, 0]
                gt    = label[i:i+1].detach().float().cpu().numpy()[0, :, :, 0]
                diff  = np.abs(recon - gt)

                def norm01(x):
                    x = x.astype(np.float32)
                    x = x - x.min()
                    return x / (x.max() + 1e-8)

                base = os.path.join(
                    args.samples_dir,
                    f"sample{saved_n:02d}_{os.path.basename(f_full)}_slice{sid:03d}"
                )
                plt.imsave(base + "_recon.png",   norm01(recon), cmap="gray")
                plt.imsave(base + "_gt.png",      norm01(gt),    cmap="gray")
                plt.imsave(base + "_absdiff.png", norm01(diff),  cmap="gray")

                saved_n += 1
        # ----------------------------

        output_up_kspace = rfft2(output_up)
        output_down_kspace = rfft2(output_down)

        diff_otherf = (output_up_kspace - output_down_kspace) * (1 - mask_under)

        recon_loss_up = criterion(output_up_kspace * mask_under, under_kspace)
        recon_loss_down = criterion(output_down_kspace * mask_under, under_kspace)
        diff_loss = criterion(diff_otherf, torch.zeros_like(diff_otherf))

        constr_loss_up = criterion(loss_layers_up[0], torch.zeros_like(loss_layers_up[0]))
        constr_loss_down = criterion(loss_layers_down[0], torch.zeros_like(loss_layers_down[0]))
        for i in range(args.num_layers - 1):
            constr_loss_up += criterion(loss_layers_up[i + 1], torch.zeros_like(loss_layers_up[i + 1]))
            constr_loss_down += criterion(loss_layers_down[i + 1], torch.zeros_like(loss_layers_down[i + 1]))

        batch_loss = recon_loss_up + recon_loss_down + 0.01 * diff_loss + 0.01 * constr_loss_up + 0.01 * constr_loss_down

        if mode == 'train':
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        else:
            # compute metrics only if we have label (val always should)
            B = label.shape[0]
            for i in range(B):
                if (mode == 'val') or has_label_list[i]:
                    psnr += psnr_slice(label[i:i+1], output_up[i:i+1])
                    ssim += ssim_slice(label[i:i+1], output_up[i:i+1])
                    metric_n += 1

        loss += batch_loss.item()

    loss /= len(dataloader)
    log.append(loss)

    if mode == 'train':
        curr_lr = optimizer.param_groups[0]['lr']
        log.append(curr_lr)
    else:
        if metric_n > 0:
            psnr /= metric_n
            ssim /= metric_n
        else:
            psnr, ssim = float('nan'), float('nan')
        log.append(psnr)
        log.append(ssim)

    # If we were auto-picking reference pairs (i.e., no --reference-pairs was given),
    # write them so the next run can reproduce the exact same slices.
    if want_save and (ref_pairs is None) and (rank == 0) and hasattr(args, "_picked_pairs"):
        out_txt = os.path.join(args.samples_dir, "picked_reference_pairs.tsv")
        with open(out_txt, "w") as f:
            for fp, sid in sorted(args._picked_pairs.items()):
                f.write(f"{fp}\t{sid}\n")
        print("Wrote reference pairs:", out_txt)


    return log


def solvers(rank, ngpus_per_node, args):
    if rank == 0:
        logger = create_logger()
        if args.gpus > 1:
            logger.info(f'Running distributed data parallel on {args.gpus} gpus.')
        else:
            logger.info('Running single-process on 1 gpu (no DDP).')

    is_distributed = (args.gpus > 1) and dist.is_available() and dist.is_initialized()
    if is_distributed:
        dist.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.world_size,
            rank=rank
        )
        torch.cuda.set_device(rank)
    else:
        # single process / single GPU: no process group
        torch.cuda.set_device(0)  

    # set initial value
    start_epoch = 0
    best_ssim = 0.0
    # model
    if args.supervised_ista:
        model = SingleNetwork(args.num_layers, rank)   # one net (supervised)
    elif args.ssdu:
        model = SingleNetwork(args.num_layers, rank)   # one net, different loss (self-supervised)
    else:
        model = ParallelNetwork(args.num_layers, rank) # two nets
    model = model.to(rank)

    if is_distributed:
        model = DDP(model, device_ids=[rank], output_device=rank)
        
    # criterion, optimizer, learning rate scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    warm_up = lambda epoch: epoch / args.warmup_epochs if epoch <= args.warmup_epochs else 1
    scheduler_wu = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_up)
    scheduler_re = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.3, patience=20)
    early_stopping = EarlyStopping(patience=50, delta=1e-5)

    # whether load checkpoint
    if args.pretrained or args.mode == 'test':
        model_path = os.path.join(args.model_save_path, 'checkpoint.pth.tar')
        if not os.path.exists(model_path):
            model_path = os.path.join(args.model_save_path, "best_checkpoint.pth.tar")
        assert os.path.isfile(model_path)
        try:
            checkpoint = torch.load(model_path, map_location=f'cuda:{rank}', weights_only=False)
        except TypeError:
            # Older PyTorch versions don't have weights_only arg
            checkpoint = torch.load(model_path, map_location=f'cuda:{rank}')
        
        start_epoch = int(checkpoint.get("epoch", 0))
        lr = float(checkpoint.get("lr", args.lr))
        args.lr = lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        best_ssim = float(checkpoint.get("best_ssim", 0.0))
        state = checkpoint.get("state_dict", checkpoint.get("model"))
        model.load_state_dict(state)

        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if checkpoint.get("scheduler_re") is not None:
            scheduler_re.load_state_dict(checkpoint["scheduler_re"])
        #if checkpoint.get("scheduler_wu") is not None:
            #scheduler_wu.load_state_dict(checkpoint["scheduler_wu"])    
        
        if start_epoch < args.warmup_epochs:
                warm_lr = args.lr * (start_epoch + 1) / args.warmup_epochs
                for pg in optimizer.param_groups:
                    pg["lr"] = warm_lr
        if rank == 0:
            logger.info('Load checkpoint at epoch {}.'.format(start_epoch))
            logger.info('Current learning rate is {}.'.format(lr))
            logger.info('Current best ssim in train phase is {}.'.format(best_ssim))
            logger.info('The model is loaded.')
    elif args.use_init_weights:
        init_weights(model, init_type=args.init_type, gain=args.gain)
        if rank == 0:
            logger.info('Initialize model with {}.'.format(args.init_type))

    # test step
    if args.mode == 'test':
        if args.samples_dir is None:
            args.samples_dir = os.path.join(os.path.dirname(args.model_save_path), "samples")

        test_set = Dataset(
            args.test_path, args.u_mask_path, args.s_mask_up_path, args.s_mask_down_path, args.test_sample_rate,
            file_list=args.test_file_list,
            ssdu=args.ssdu, seed=args.seed, ssdu_theta_frac=args.ssdu_theta_frac
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
            drop_last=False
        )
        if rank == 0:
            logger.info('The size of test dataset is {}.'.format(len(test_set)))
            logger.info('Now testing {}.'.format(args.exp_name))
        model.eval()
        with torch.no_grad():
            test_log = []
            start_time = time.time()
            test_log = forward('test', rank, model, test_loader, criterion, optimizer, test_log, args)
            test_time = time.time() - start_time
        # test information
        test_loss = test_log[0]
        test_psnr = test_log[1]
        test_ssim = test_log[2]
        if rank == 0:
            logger.info('time:{:.5f}s\ttest_loss:{:.7f}\ttest_psnr:{:.5f}\ttest_ssim:{:.5f}'.format(test_time, test_loss, test_psnr, test_ssim))
        return       

    # training step
    train_set = Dataset(
        args.train_path, args.u_mask_path, args.s_mask_up_path, args.s_mask_down_path, args.train_sample_rate,
        seed=args.seed, ssdu=args.ssdu, ssdu_theta_frac=args.ssdu_theta_frac, file_list=args.train_file_list
    )

    # Only use DistributedSampler if we actually initialized distributed
    print("is_distributed:", is_distributed, "dist initialized:", dist.is_initialized())
    train_sampler = DistributedSampler(train_set) if is_distributed else None

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        pin_memory=True,
        sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=True
    )

    val_set = Dataset(
        args.val_path, args.u_mask_path, args.s_mask_up_path, args.s_mask_down_path, args.val_sample_rate,
        file_list=args.val_file_list
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=False
    )

    if rank == 0:
        logger.info('The size of training dataset and validation dataset is {} and {}, respectively.'.format(len(train_set), len(val_set)))
        logger.info('Now training {}.'.format(args.exp_name))
        writer = SummaryWriter(args.loss_curve_path)

    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        if args.ssdu and hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_log = [epoch]
        epoch_start_time = time.time()

        model.train()
        train_log = forward('train', rank, model, train_loader, criterion, optimizer, train_log, args)

        model.eval()
        with torch.no_grad():
            train_log = forward('val', rank, model, val_loader, criterion, optimizer, train_log, args)

        epoch_time = time.time() - epoch_start_time

        # train information
        epoch = train_log[0]
        train_loss = train_log[1]
        lr = optimizer.param_groups[0]["lr"]
        val_loss = train_log[3]
        val_psnr = train_log[4]
        val_ssim = train_log[5]


        is_best = val_ssim > best_ssim
        best_ssim = max(val_ssim, best_ssim)

        if rank == 0:
            logger.info('epoch:{:<8d}time:{:.5f}s\tlr:{:.8f}\ttrain_loss:{:.7f}\tval_loss:{:.7f}\tval_psnr:{:.5f}\t'
                        'val_ssim:{:.5f}'.format(epoch, epoch_time, lr, train_loss, val_loss, val_psnr, val_ssim))
            writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)

            # --- SAVE CHECKPOINT (resumable) ---
            model_state = (model.module.state_dict() if hasattr(model, "module") else model.state_dict())

            checkpoint = {
                "epoch": epoch,                 # epoch index that just finished
                "lr": lr,
                "best_ssim": best_ssim,

                # NEW standard keys
                "state_dict": model_state,
                "optimizer": optimizer.state_dict(),

                # Optional but recommended: schedulers (so LR schedule resumes properly)
                "scheduler_re": scheduler_re.state_dict() if scheduler_re is not None else None,
                "scheduler_wu": scheduler_wu.state_dict() if scheduler_wu is not None else None,

                # Keep legacy key so older code paths still work
                "model": model_state,
            }

            if not os.path.exists(args.model_save_path):
                os.makedirs(args.model_save_path)

            model_path = os.path.join(args.model_save_path, 'checkpoint.pth.tar')
            best_model_path = os.path.join(args.model_save_path, 'best_checkpoint.pth.tar')
            torch.save(checkpoint, model_path)
            if is_best:
                shutil.copy(model_path, best_model_path)

        # scheduler
        if epoch <= args.warmup_epochs:
            scheduler_wu.step(epoch)
        scheduler_re.step(val_ssim)

        early_stopping(val_ssim, loss=False)
        if early_stopping.early_stop:
            if rank == 0:
                logger.info('The experiment is early stop!')
            break

    if rank == 0:
        writer.close()
    

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        
    return



def main():
    args = parser.parse_args()
    print("save_samples:", getattr(args, "save_samples", None),
      "num_save_samples:", getattr(args, "num_save_samples", None),
      "samples_dir:", getattr(args, "samples_dir", None),
      "ssdu:", getattr(args, "ssdu", None))
    if args.supervised_ista and args.ssdu:
        raise ValueError("Choose one: --ssdu OR --supervised-ista (not both).")
    args.world_size = args.nodes * args.gpus
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.gpus == 1:
        solvers(0, 1, args)
    else:
        torch.multiprocessing.spawn(solvers, nprocs=args.gpus, args=(args.gpus, args))


if __name__ == '__main__':
    main()
