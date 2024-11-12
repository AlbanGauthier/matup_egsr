import torch
import numpy as np
import os
from tqdm import tqdm
from einops import rearrange

from . import data_utils

def eval_model(args, model, img_dataset, device) :

	padding = (args.tile_width - 1) // 2

	model.eval()
	with torch.no_grad():

		data_vec = img_dataset.get_svbrdf_init().clone().to(device)

		svbrdf_in = rearrange(data_vec, 'h w c -> 1 c h w')
		mlp_delta = model(svbrdf_in)
		mlp_delta = rearrange(mlp_delta, '1 c h w -> h w c')

		mlp_out = data_vec[padding:-padding, padding:-padding, ...] + mlp_delta

		position3D = img_dataset.get_position3D()[
			padding:-padding, padding:-padding][..., None].to(device)

		maps_eval = data_utils.maps_from_data(mlp_out, position3D)
		maps_eval = data_utils.get_renorm_clamped_maps(maps_eval)

	model.train()
	return maps_eval