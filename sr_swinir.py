
import os
import sys
import cv2
import time
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "sr_modules", "SwinIR"))

from sr_modules.SwinIR.main_test_swinir import define_model, setup, get_image_pair, test

def main_upsample_swinir(args):

    args.task       = 'real_sr'
    args.scale      = 4
    args.folder_gt  = None
    args.tile       = None
    args.tile_overlap = 32
    args.model_url  = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"

    args.large_model = True
    args.model_path = "./pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"

    args.folder_lq = args.render_dir
    args.out_dir = args.upsamplings_dir

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        print("please download model at: ", args.model_url)
        exit(0)

    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    save_dir = args.out_dir
    os.makedirs(save_dir, exist_ok=True)

    files_list = sorted(glob.glob(os.path.join(folder, '*')))
    nb_renders = len(files_list)

    prog_bar = tqdm(total=nb_renders, leave=False)

    image_batch = torch.zeros((nb_renders, 3, args.material_size, args.material_size))

    tic = time.perf_counter()

    for idx, path in enumerate(files_list):

        # read image
        _, img_lq, _ = get_image_pair(args, path)  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0)  # CHW-RGB to NCHW-RGB
        image_batch[idx] = img_lq

        prog_bar.update(1)

    prog_bar.close()

    image_batch = image_batch.to(device)

    output_list = []

    prog_bar = tqdm(total=args.up_batch_size, leave=False)

    # pad input image to be a multiple of window_size
    _, _, h_old, w_old = image_batch.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    image_batch = torch.cat([image_batch, torch.flip(image_batch, [2])], 2)[:, :, :h_old + h_pad, :]
    image_batch = torch.cat([image_batch, torch.flip(image_batch, [3])], 3)[:, :, :, :w_old + w_pad]

    # inference
    with torch.no_grad():
        for i in range(args.up_batch_size):
            output = test(image_batch[(nb_renders // args.up_batch_size) *i:(nb_renders // args.up_batch_size) *(i+1)], model, args, window_size)
            output = output[..., :h_old * args.scale, :w_old * args.scale]
            output_list.append(output)
            prog_bar.update(1)

    prog_bar.close()

    toc = time.perf_counter()
    print(f"Upsampling took {toc - tic:0.4f} seconds")
    tic = toc

    for batch_idx, img_vec in enumerate(output_list):
        for i in range(nb_renders // args.up_batch_size):
            # save image
            output_ = img_vec[i].data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if output_.ndim == 3:
                output_ = np.transpose(output_[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            output_ = (output_ * 255.0).round().astype(np.uint8)  # float32 to uint8
            cv2.imwrite(f'{save_dir}/render_{i + batch_idx*(nb_renders // args.up_batch_size):04d}_SwinIR.png', output_)
            
    toc = time.perf_counter()
    print(f"Saving images took {toc - tic:0.4f} seconds")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--render_dir", 	    type=str, 	default="./data/victorian_wood/radiance_LR/") 
    parser.add_argument("--upsamplings_dir", 	type=str,	default="./data/victorian_wood/radiance_SwinIR/")

    parser.add_argument('--material_size',  type=int, 	default=128)
    parser.add_argument("--up_batch_size",  type=int, 	default=10, help="nb of parallel upsamplings, must divide nb_renders")

    args = parser.parse_args()
    main_upsample_swinir(args)