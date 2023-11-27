# System / Python
import os
import argparse
import logging
import random
import shutil
import time

import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import cv2
# PyTorch

import torch.fft
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
# Custom
from net import *
from dataset import *
from mri_tools import *
from utils import psnr_slice, ssim_slice, get_cos_similar_matrix, get_cos_similar, Recorder
import logging


class similarity_loss(nn.Module):
    def __init__(self):
        super(similarity_loss, self).__init__()

    def forward(self, x, y):
        s = torch.exp(get_cos_similar(x, y))
        return -torch.log(s/(s+torch.tensor(0.05)))


def forward_parcel(mode, rank, model, dataloader, criterion, optimizer, log, mlp, args):
    assert mode in ['train', 'val', 'test']
    loss, psnr, ssim = 0.0, 0.0, 0.0
    recorder = Recorder(os.path.join("./results", args.exp_name))
    t = tqdm(dataloader, desc=mode + 'ing',
             total=int(len(dataloader))) if rank == 0 else dataloader
    for iter_num, data_batch in enumerate(t):
        # full_kspace = data_batch[0].to(rank, non_blocking=True)
        # csm = data_batch[1].to(rank, non_blocking=True)
        # mask_under = data_batch[2].to(rank, non_blocking=True)
        # mask_net_up = data_batch[3].to(rank, non_blocking=True)
        # mask_net_down = data_batch[4].to(rank, non_blocking=True)
        
        full_kspace = data_batch[0].to(rank, non_blocking=True)
        csm = data_batch[1].to(rank, non_blocking=True)
        mask_under = data_batch[2][0].to(rank, non_blocking=True)
        fnames = data_batch[3]
        slice_id = data_batch[4]
        mask_net_up = data_batch[5][0].to(rank, non_blocking=True)
        mask_net_down = data_batch[5][1].to(rank, non_blocking=True)
        # fname = data_batch[5]
        # slice_id = data_batch[6]

        label = torch.sum(ifft2(full_kspace) * torch.conj(csm), dim=1)

        under_img = At(full_kspace, csm, mask_under)
        under_img = torch.view_as_real(
            under_img).permute(0, 3, 1, 2).contiguous()

        net_img_up = At(full_kspace, csm, mask_net_up)
        net_img_up = torch.view_as_real(
            net_img_up).permute(0, 3, 1, 2).contiguous()

        net_img_down = At(full_kspace, csm, mask_net_down)
        net_img_down = torch.view_as_real(
            net_img_down).permute(0, 3, 1, 2).contiguous()

        if mode == 'test':
            net_img_up = net_img_down = under_img
            mask_net_up = mask_net_down = mask_under

        if mode == 'test':
            with torch.no_grad():
                start_time = time.time()
                output_up, output_down = model(
                    net_img_up, mask_net_up, net_img_down, mask_net_down, csm)
                recorder.total_time += time.time()-start_time
        else:
            output_up, output_down = model(
                net_img_up, mask_net_up, net_img_down, mask_net_down, csm)

        output_up = torch.view_as_complex(
            output_up.permute(0, 2, 3, 1).contiguous())
        output_down = torch.view_as_complex(
            output_down.permute(0, 2, 3, 1).contiguous())

        output_up_kspace = fft2(output_up[:, None, ...] * csm)
        output_down_kspace = fft2(output_down[:, None, ...] * csm)

        recon_up_masked = At(output_up_kspace, csm, mask_under)
        recon_down_masked = At(output_down_kspace, csm, mask_under)

        # undersampled calibration loss
        under_img = torch.view_as_complex(
            under_img.permute(0, 2, 3, 1).contiguous())
        recon_loss_up = criterion(
            torch.abs(recon_up_masked), torch.abs(under_img))
        recon_loss_down = criterion(
            torch.abs(recon_down_masked), torch.abs(under_img))

        # data consistency
        e_up_kspace = output_up_kspace * \
            (1-mask_under)[:, None, ...] + \
            full_kspace * mask_under[:, None, ...]
        dc_up_out = torch.sum(ifft2(e_up_kspace) * torch.conj(csm), dim=1)

        e_down_kspace = output_down_kspace * \
            (1-mask_under)[:, None, ...] + \
            full_kspace * mask_under[:, None, ...]
        dc_down_out = torch.sum(ifft2(e_down_kspace) * torch.conj(csm), dim=1)

        # reconstructed calibration loss
        dc_up_loss = criterion(torch.abs(dc_up_out), torch.abs(output_up))
        dc_down_loss = criterion(
            torch.abs(dc_down_out), torch.abs(output_down))

        # constrast_loss
        cl = similarity_loss()
        s = cl(mlp(torch.abs(output_up)), mlp(torch.abs(output_down)))

        batch_loss = recon_loss_up + recon_loss_down + \
            0.01*dc_up_loss + 0.01*dc_down_loss + 0.1*s

        f_output = (output_up + output_down)/2.0
        if mode == 'train':
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        else:
            output_path = os.path.join("./results", args.exp_name, "samples")
            os.makedirs(output_path, exist_ok=True)
            output_fnames = []
            for i, fname in enumerate(fnames):
                bname = os.path.basename(fname)
                if bname.endswith(".h5"):
                    bname = bname[:-3]+f"_{data_batch[6][i]}.png"
                output_fnames.append(os.path.join(
                    output_path, "rec_"+bname.replace("png", "npy")))
            psnr += psnr_slice(label, f_output)
            ssim += ssim_slice(label, f_output)
            recorder.submit(label, f_output, output_fnames)
        loss += batch_loss.item()
    loss /= len(dataloader)
    log.append(loss)
    if mode == 'train':
        curr_lr = optimizer.param_groups[0]['lr']
        log.append(curr_lr)
    else:
        psnr /= len(dataloader)
        ssim /= len(dataloader)
        log.append(psnr)
        log.append(ssim)
        recorder.export()
    return log


def forward_parcel_multi(mode, rank, model, dataloader, criterion, optimizer, log, mlp, args):
    assert mode in ['train', 'val', 'test']
    loss, psnr, ssim = 0.0, 0.0, 0.0
    recorder = Recorder(os.path.join("./results", args.exp_name))
    t = tqdm(dataloader, desc=mode + 'ing',
             total=int(len(dataloader))) if rank == 0 else dataloader
    for iter_num, data_batch in enumerate(t):
        full_kspace = data_batch[0].to(rank, non_blocking=True)
        csm = data_batch[1].to(rank, non_blocking=True)
        mask_under_list = [i.to(rank, non_blocking=True) for i in data_batch[2]]
        fnames = data_batch[3]
        slice_id = data_batch[4]
        remask_list=[i.to(rank, non_blocking=True) for i in data_batch[5]]

        label = torch.sum(ifft2(full_kspace) * torch.conj(csm), dim=1)
        
        mask_reunder_list=[];net_img_reunder_list=[];csm_list=[]
        for mask_under,remask in zip(mask_under_list,remask_list):
            under_img = At(full_kspace, csm, mask_under)
            under_img = torch.view_as_real(
                under_img).permute(0, 3, 1, 2).contiguous()
            mask_net_reunder=mask_under*remask

            net_img_reunder = At(full_kspace, csm, mask_net_reunder)
            net_img_reunder = torch.view_as_real(
                net_img_reunder).permute(0, 3, 1, 2).contiguous()

            if mode == 'test':
                net_img_reunder = under_img
                mask_net_reunder = mask_under
            mask_reunder_list.append(mask_net_reunder)
            net_img_reunder_list.append(net_img_reunder)
            csm_list.append(csm)

        if mode == 'test':
            with torch.no_grad():
                start_time = time.time()
                output_list=model(net_img_reunder_list,mask_reunder_list,csm_list)
                recorder.total_time += time.time()-start_time
        else:
            output_list=model(net_img_reunder_list,mask_reunder_list,csm_list)
            
        batch_loss=0.;cl = similarity_loss()
        f_output = sum(output_list).detach()/len(output_list)
        for output,mask_under,csm in zip(output_list,mask_under_list,csm_list):
            output=torch.view_as_complex(
                output.permute(0, 2, 3, 1).contiguous())
            output_kspace = fft2(output[:, None, ...] * csm)
            recon_masked=At(output_kspace, csm, mask_under)

            # undersampled calibration loss
            under_img = At(full_kspace, csm, mask_under)
            recon_loss = criterion(
                torch.abs(recon_masked), torch.abs(under_img))

            # data consistency
            e_kspace = output_kspace * \
                (1-mask_under)[:, None, ...] + \
                full_kspace * mask_under[:, None, ...]
            dc_out = torch.sum(ifft2(e_kspace) * torch.conj(csm), dim=1)

            # reconstructed calibration loss
            dc_loss = criterion(torch.abs(dc_out), torch.abs(output))
            batch_loss=batch_loss+recon_loss+0.01*dc_loss
            # print(f"recon_loss:{recon_loss.item()}\tdc_loss:{dc_loss.item()}",end="\t")

            # # constrast_loss
            # batch_loss=batch_loss+0.1*cl(mlp(torch.abs(output)), mlp(torch.abs(f_output)))
        for output_index,output in enumerate(output_list):
            for output2_index, output2 in enumerate(output_list):
                if output_index>=output2_index:
                    continue
                cl_loss=cl(mlp(torch.abs(output)), mlp(torch.abs(output2)))
                batch_loss=batch_loss+0.1*cl_loss
                # print(f"cl_loss{cl_loss.item()}")
            
        f_output=torch.view_as_complex(
            f_output.permute(0, 2, 3, 1).contiguous())
        if mode == 'train':
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        else:
            output_path = os.path.join("./results", args.exp_name, "samples")
            os.makedirs(output_path, exist_ok=True)
            output_fnames = []
            for i, fname in enumerate(fnames):
                bname = os.path.basename(fname)
                if bname.endswith(".h5"):
                    bname = bname[:-3]+f"_{data_batch[6][i]}.png"
                output_fnames.append(os.path.join(
                    output_path, "rec_"+bname.replace("png", "npy")))
                for n,o in enumerate(output_list):
                    np.save(output_fnames[i].replace("rec",str(n)),o[i].cpu().numpy())
            psnr += psnr_slice(label, f_output)
            ssim += ssim_slice(label, f_output)
            recorder.submit(label, f_output, output_fnames)
        loss += batch_loss.item()
    loss /= len(dataloader)
    log.append(loss)
    if mode == 'train':
        curr_lr = optimizer.param_groups[0]['lr']
        log.append(curr_lr)
    else:
        psnr /= len(dataloader)
        ssim /= len(dataloader)
        log.append(psnr)
        log.append(ssim)
        recorder.export()
    return log


def forward_dccnn_sup(mode, rank, model, dataloader, criterion, optimizer, log, mlp, args):
    print("Performing forward_dccnn_sup")
    assert mode in ['train', 'val', 'test']
    loss, psnr, ssim = 0.0, 0.0, 0.0
    recorder = Recorder(os.path.join("./results", args.exp_name))
    t = tqdm(dataloader, desc=mode + 'ing',
             total=int(len(dataloader))) if rank == 0 else dataloader
    for iter_num, data_batch in enumerate(t):
        full_kspace = data_batch[0].to(rank, non_blocking=True)
        csm = data_batch[1].to(rank, non_blocking=True)
        mask_under = data_batch[2].to(rank, non_blocking=True)
        mask_net_up = data_batch[3].to(rank, non_blocking=True)
        mask_net_down = data_batch[4].to(rank, non_blocking=True)
        # fname = data_batch[5]
        # slice_id = data_batch[6]

        label = torch.sum(ifft2(full_kspace) * torch.conj(csm), dim=1)

        input_kspace = full_kspace * mask_under[:, None, ...]
        if mode == 'test':
            with torch.no_grad():
                start_time = time.time()
                output = model(input_kspace, csm, mask_under)
                recorder.total_time += time.time()-start_time
        else:
            output = model(input_kspace, csm, mask_under)
        output = torch.sum(ifft2(output) * torch.conj(csm), dim=1)

        # loss
        recon_loss = criterion(torch.abs(output), torch.abs(label))

        batch_loss = recon_loss

        f_output = output

        if mode == 'train':
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        else:
            fnames = data_batch[5]
            output_path = os.path.join("./results", args.exp_name, "samples")
            os.makedirs(output_path, exist_ok=True)
            output_fnames = []
            for i, fname in enumerate(fnames):
                bname = os.path.basename(fname)
                if bname.endswith(".h5"):
                    bname = bname[:-3]+f"_{data_batch[6][i]}.png"
                output_fnames.append(os.path.join(
                    output_path, "rec_"+bname.replace("png", "npy")))
            psnr += psnr_slice(label, f_output)
            ssim += ssim_slice(label, f_output)
            recorder.submit(label, f_output, output_fnames)
        loss += batch_loss.item()
    loss /= len(dataloader)
    log.append(loss)
    if mode == 'train':
        curr_lr = optimizer.param_groups[0]['lr']
        log.append(curr_lr)
    else:
        psnr /= len(dataloader)
        ssim /= len(dataloader)
        log.append(psnr)
        log.append(ssim)
        recorder.export()
    return log


def forward_dccnn_distill_parcel(mode, rank, model, dataloader, criterion, optimizer, log, mlp, args):
    print("Performing forward_dccnn_distill_parcel")
    if mode == "test":
        return forward_dccnn_sup(mode, rank, model, dataloader, criterion, optimizer, log, mlp, args)
    assert mode in ['train', 'val', 'test']
    loss, psnr, ssim = 0.0, 0.0, 0.0
    recorder = Recorder(os.path.join("./results", args.exp_name))
    t = tqdm(dataloader, desc=mode + 'ing',
             total=int(len(dataloader))) if rank == 0 else dataloader

    def get_average():
        if hasattr(args.distillee_params, "average") and args.distillee_params.average:
            return args.distillee_params.average
        else:
            sum_se = []
            for iter_num, data_batch in enumerate(t):
                with torch.no_grad():
                    full_kspace = data_batch[0].to(rank, non_blocking=True)
                    csm = data_batch[1].to(rank, non_blocking=True)
                    mask_under = data_batch[2].to(rank, non_blocking=True)
                    mask_net_up = data_batch[3].to(rank, non_blocking=True)
                    mask_net_down = data_batch[4].to(rank, non_blocking=True)
                    fname = data_batch[5]
                    # slice_id = data_batch[6]

                    label = torch.sum(ifft2(full_kspace) *
                                      torch.conj(csm), dim=1)

                    under_img = At(full_kspace, csm, mask_under)
                    under_img = torch.view_as_real(
                        under_img).permute(0, 3, 1, 2).contiguous()
                    d_output_up, d_output_down = args.distillee_params.model(
                        under_img, mask_under, under_img, mask_under, csm)
                    d_f_output = (d_output_up + d_output_down)/2.0
                    d_f_output = torch.view_as_complex(
                        d_f_output.permute(0, 2, 3, 1).contiguous())
                    sum_se.append(torch.stack([torch.abs(torch.view_as_complex(d_output_up.permute(0, 2, 3, 1).contiguous())),
                                              torch.abs(torch.view_as_complex(d_output_down.permute(0, 2, 3, 1).contiguous()))])
                                  .var(dim=0).sum(dim=(-2, -1)))
            args.distillee_params.average = torch.concat(sum_se, dim=0).mean()
            print(args.distillee_params.average)
            return args.distillee_params.average
    for iter_num, data_batch in enumerate(t):
        full_kspace = data_batch[0].to(rank, non_blocking=True)
        csm = data_batch[1].to(rank, non_blocking=True)
        mask_under = data_batch[2].to(rank, non_blocking=True) 
        mask_net_up = data_batch[3].to(rank, non_blocking=True)
        mask_net_down = data_batch[4].to(rank, non_blocking=True)
        # fname = data_batch[5]
        # slice_id = data_batch[6]

        label = torch.sum(ifft2(full_kspace) * torch.conj(csm), dim=1)

        under_img = At(full_kspace, csm, mask_under)
        under_img = torch.view_as_real(
            under_img).permute(0, 3, 1, 2).contiguous()

        input_kspace = full_kspace * mask_under[:, None, ...]

        output = model(input_kspace, csm, mask_under)
        output = torch.sum(ifft2(output) * torch.conj(csm), dim=1)

        # distill generation
        if mode == "train" or mode == "val":
            with torch.no_grad():
                d_output_up, d_output_down = args.distillee_params.model(
                    under_img, mask_under, under_img, mask_under, csm)
                d_f_output = (d_output_up + d_output_down)/2.0
                d_f_output = torch.view_as_complex(
                    d_f_output.permute(0, 2, 3, 1).contiguous())
            if hasattr(args.distillee_params, "sample_reweighting"):
                get_average()
                weighting = (args.distillee_params.average/torch.stack([torch.abs(torch.view_as_complex(d_output_up.permute(0, 2, 3, 1).contiguous())),
                                                                        torch.abs(torch.view_as_complex(d_output_down.permute(0, 2, 3, 1).contiguous()))])
                             .var(dim=0).sum(dim=(-2, -1))).unsqueeze(1).unsqueeze(1).clip(1/3, 3)
                distill_loss = criterion(
                    torch.abs(output)*weighting, torch.abs(d_f_output)*weighting)
            else:
                distill_loss = criterion(
                    torch.abs(output), torch.abs(d_f_output))

            batch_loss = distill_loss
        else:
            batch_loss = torch.tensor(0.)

        f_output = output

        if mode == 'train':
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        else:
            fnames = data_batch[5]
            output_path = os.path.join("./results", args.exp_name, "samples")
            os.makedirs(output_path, exist_ok=True)
            output_fnames = []
            for i, fname in enumerate(fnames):
                bname = os.path.basename(fname)
                if bname.endswith(".h5"):
                    bname = bname[:-3]+f"_{data_batch[6][i]}.png"
                output_fnames.append(os.path.join(
                    output_path, "rec_"+bname.replace("png", "npy")))
            psnr += psnr_slice(label, f_output)
            ssim += ssim_slice(label, f_output)
            recorder.submit(label, f_output, output_fnames)
        loss += batch_loss.item()
    loss /= len(dataloader)
    log.append(loss)
    if mode == 'train':
        curr_lr = optimizer.param_groups[0]['lr']
        log.append(curr_lr)
    else:
        psnr /= len(dataloader)
        ssim /= len(dataloader)
        log.append(psnr)
        log.append(ssim)
        recorder.export()
    return log
