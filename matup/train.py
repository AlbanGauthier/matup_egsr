import torch
import time
import datetime
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

from . import render_utils
from . import data_utils
from . import testing

def loss_fn(mlp_renders, GT_renders, args, loss_MSE, maps_var, mid_sample):

	diff_vec = mlp_renders - GT_renders.squeeze()

	# l_2,1 norm
	l21_loss = torch.sum(torch.sqrt(
		torch.clamp(torch.sum(
			torch.pow(diff_vec, 2.0), dim=1), # L2 RGB norm
				min=1e-15))) / (args.batch_size * args.nb_lights) # renorm

	# metallic regularization
	svbrdf_mid_sample = mid_sample.squeeze(1)
	reg_loss = args.alpha_reg * loss_MSE(
			maps_var[1].view(-1), 
			svbrdf_mid_sample[..., 3:4].view(-1))
		
	return l21_loss + reg_loss


def mlp_forward_pass(svbrdf, model, args, position3D, GT_renders):
	
	# prepare for MLP
	svbrdf_in = svbrdf.permute(0, 3, 1, 2)
	
	# MLP forward
	mlp_delta = model(svbrdf_in)

	mlp_delta = mlp_delta.permute(0, 2, 3, 1)
	svbrdf_mid_sample = svbrdf[:,
		args.tile_width // 2: args.tile_width // 2 + 1,
		args.tile_width // 2: args.tile_width // 2 + 1, :]

	# add MLP output to texel center
	mlp_out = svbrdf_mid_sample + mlp_delta

	mlp_out 	= mlp_out.squeeze(1)
	position3D 	= position3D.squeeze(1)
	# GT_renders 	= (GT_renders / 255.0).float()
	GT_renders 	= GT_renders.squeeze(1)

	maps_var = data_utils.maps_from_data(mlp_out, position3D)
	maps_var = data_utils.get_renorm_clamped_maps(maps_var)

	return maps_var, GT_renders, svbrdf_mid_sample


def train_model(args, model, optimizer, train_dataset, data_loader, loss_MSE, wi_tab, device):

	svbrdfs = [
		torch.empty((args.batch_size, args.tile_width, args.tile_width, 7), device="cuda"), 
		torch.empty((args.batch_size, args.tile_width, args.tile_width, 7), device="cuda")]
	GT_renders = [
		torch.empty((args.batch_size, 1, 1, 3, args.nb_lights), device="cuda"), 
		torch.empty((args.batch_size, 1, 1, 3, args.nb_lights), device="cuda")]
	position3Ds = [
		torch.empty((args.batch_size, 1, 1, 3) , device="cuda"), 
		torch.empty((args.batch_size, 1, 1, 3) , device="cuda")]

	its = 0
	start = 0
	warmup = 4

	data_ready 		= [torch.cuda.Event(), torch.cuda.Event()]
	training_ready 	= [torch.cuda.Event(), torch.cuda.Event()] 
	streams			= [torch.cuda.Stream(), torch.cuda.Stream()]
	graphs 			= [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]

	loss_val = torch.tensor([0.0]).cuda().float()

	for epoch_id in range(start + 1, args.nb_epochs + start + 1):

		model.train()
		loss_val[0] = 0
		prog_bar = tqdm(total=len(data_loader), leave=False)

		for svbrdf_, GT_sample_, position3D_ in data_loader:

			if its <= 2 * warmup:

				j = 0 if its <= warmup else 1 

				svbrdfs[j].copy_(svbrdf_)
				GT_renders[j].copy_(GT_sample_)
				position3Ds[j].copy_(position3D_)

				if its < ((j+1) * warmup):
					with torch.cuda.stream(streams[0]):
						optimizer.zero_grad(True)

						maps, data, mid = mlp_forward_pass(svbrdfs[j], model, args, position3Ds[j], GT_renders[j])
						render_var = render_utils.batch_render_tn_explicit(*maps, wi_tab)
						loss = loss_fn(render_var, data, args, loss_MSE, maps, mid)
						loss.backward()
						optimizer.step()
						loss_val += loss
					torch.cuda.current_stream().wait_stream(streams[0])
				else:
					with torch.cuda.graph(graphs[j]):
						optimizer.zero_grad(True)

						maps, data, mid = mlp_forward_pass(svbrdfs[j], model, args, position3Ds[j], GT_renders[j])
						render_var = render_utils.batch_render_tn_explicit(*maps, wi_tab)
						loss = loss_fn(render_var, data, args, loss_MSE, maps, mid)
						loss.backward()
						optimizer.step()
						loss_val += loss
					graphs[j].replay()
			else:

				j = 0 if its % 2 == 0 else 1

				with torch.cuda.stream(streams[0]):
					training_ready[j].wait()
					svbrdfs[j].copy_(svbrdf_)
					GT_renders[j].copy_(GT_sample_)
					position3Ds[j].copy_(position3D_)
					data_ready[j].record()

				with torch.cuda.stream(streams[1]):					
					data_ready[j].wait()
					graphs[j].replay()
					training_ready[j].record()

			if its % 128 == 0:
				prog_bar.update(128)
				 
			its += 1

		prog_bar.close()

		torch.cuda.synchronize()

		loss_value = loss_val.item() / len(data_loader)

		print("loss", epoch_id, ":", loss_value)

		if (epoch_id % args.save_interval == 0):
			maps_eval = testing.eval_model(args, model, train_dataset, device)
			data_utils.output_maps_from_list(maps_eval, args, suffix=str(epoch_id).zfill(4), pad_output=True)

	maps_eval = testing.eval_model(args, model, train_dataset, device)

	return maps_eval
