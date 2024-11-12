
import os
import sys
import argparse

sys.path.append(os.path.join(os.getcwd(), "sr_modules", "SRFlow", "code"))

import natsort, glob, pickle, torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from sr_modules.SRFlow.code.test import load_model, imread

def find_files(wildcard): return natsort.natsorted(glob.glob(wildcard, recursive=True))

def imshow(array):
    Image.fromarray(array)

def pickleRead(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
# Convert to tensor
def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255

# convert to image
def rgb(t): return (np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(np.uint8)

def main_upsample_srflow(args):

    conf_path = 'sr_modules/SRFlow_DF2K_4X_custom.yml'
    model, _ = load_model(conf_path)

    files_list = sorted(glob.glob(os.path.join(args.render_dir, '*')))
    nb_renders = len(files_list)

    image_batch = torch.zeros((nb_renders, 3, args.material_size, args.material_size))

    for idx, path in enumerate(files_list):
        # read image
        lq = imread(path)
        image_batch[idx] = t(lq)

    output_list = []
    prog_bar = tqdm(total=args.up_batch_size, leave=False)

    # inference
    with torch.no_grad():
        for i in range(nb_renders // args.up_batch_size):
            output = model.get_sr(lq=image_batch[args.up_batch_size*i:args.up_batch_size*(i+1)], heat=args.temperature)
            output_list.append(output)
            prog_bar.update(1)

    prog_bar.close()

    os.makedirs(args.upsamplings_dir, exist_ok=True)

    for batch_idx, img_vec in enumerate(output_list):
        for i in range(args.up_batch_size):
            if i + batch_idx * args.up_batch_size < nb_renders:
                # save image
                Image.fromarray(rgb(img_vec[i])).save(
                    os.path.join(args.upsamplings_dir, 
                                 f"{i + batch_idx*(args.up_batch_size):04d}_SRFlow05.png"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--render_dir", 	    type=str, 	default="./data/victorian_wood/radiance_LR/") 
    parser.add_argument("--upsamplings_dir", 	type=str,	default="./data/victorian_wood/radiance_SRFlow/")

    parser.add_argument('--temperature',    type=float, default=0.5)
    parser.add_argument('--material_size',  type=int, 	default=128)
    parser.add_argument("--up_batch_size",  type=int, 	default=10, help="nb of parallel upsamplings, must divide nb_renders")

    args = parser.parse_args()
    main_upsample_srflow(args)