import torch
import os
import cv2
import numpy as np

from . import renderer
from . import render_utils
from . import file_utils

def compute_2D_position_grid(size):
	range_vec = 2 * (np.arange(size) / size + 1 / (2 * size)) - 1
	X, Y = np.meshgrid(range_vec, range_vec)
	coords = torch.cat((
		torch.from_numpy(X).unsqueeze(-1),
		torch.from_numpy(Y).unsqueeze(-1)), dim=-1)
	return coords


def compute_3D_position_grid(size):
	range_vec = 2 * (np.arange(size) / size + 1 / (2 * size)) - 1
	X, Y = np.meshgrid(range_vec, range_vec)
	zeros = torch.zeros((size,size,1))
	coords = torch.cat((
		torch.from_numpy(X).unsqueeze(-1),
		torch.from_numpy(Y).unsqueeze(-1), zeros), dim=-1)
	return coords


def average_patches(in_vec, ks = 2, dim1 = 0, dim2 = 1):
	assert in_vec.shape[dim1] == in_vec.shape[dim2]
	in_vec = in_vec.unfold(dim1, ks, ks).unfold(dim2, ks, ks) 
	in_vec = torch.sum(in_vec, dim=-1)
	in_vec = torch.sum(in_vec, dim=-1)
	in_vec = in_vec / (ks * ks)
	return in_vec


def average_maps(map_list, ks):
	for i, map in enumerate(map_list):
		map_list[i] = average_patches(map, ks)
	map_list[0] = torch_norm(map_list[0]) # normal
	return map_list


def normalize_np_norm(normalmap):
	normalmap = 2.0 * normalmap - 1.0
	# Compute the norm of each vector along the last axis (axis=2)
	norms = np.linalg.norm(normalmap, axis=2, keepdims=True)
	# Normalize each vector by dividing by its norm, handling potential division by zero
	normalized_normals = np.divide(normalmap, norms, where=(norms != 0))
	return 0.5 * normalized_normals + 0.5


def torch_norm(tensor, dimToNorm=2, eps=1e-20):
	assert tensor.shape[dimToNorm] == 3
	length = torch.sqrt(torch.clamp(
		torch.sum(torch.square(tensor), 
			axis=dimToNorm, keepdim=True), min=eps))
	return torch.div(tensor, length)


def normal_from_data_vector(data_vec):
	
	normals = render_utils.compute_normal_from_slopes(data_vec[..., 0:2])

	if normals.ndim == 2:
		normals = torch_norm(normals, dimToNorm=1)
	else:
		normals = torch_norm(normals)
	
	return normals


def get_renorm_clamped_maps(in_maps):

	# albedo
	in_maps[0] = torch.clamp(in_maps[0], min=0, max=1)
	# metallic
	in_maps[1] = torch.clamp(in_maps[1], min=0, max=1)
	# normal
	in_maps[2] = torch_norm(in_maps[2], dimToNorm=-1)
	# roughness
	in_maps[3] = torch.clamp(in_maps[3], min=renderer.Renderer.min_alpha, max=1)

	if in_maps[0].ndim == 2:
		file_utils.exit_with_message("error in get_renorm_clamped_maps")

	return in_maps


def maps_from_data(data_vec, position3D):
	"""Return dataloader like map order"""
	ret_list = []
	# albedo
	ret_list.append(data_vec[..., :3])
	# metallic
	ret_list.append(data_vec[..., 3:4])
	# normal
	ret_list.append(normal_from_data_vector(data_vec[..., 4:6]))
	# roughness
	ret_list.append(data_vec[..., 6:7])
	# position2D + height
	ret_list.append(position3D)
	return ret_list


def output_maps_from_list(
	maps, args, suffix = "", prefix = "", pad_output=False, output_map_size=512):

	# light position for eval
	eval_light = torch.from_numpy(np.array([[0.276172, 0.276172, 0.920575]])).float()
	eval_light = torch.reshape(eval_light, (1, 3)).to(renderer.Renderer.z_vector.device)

	render = render_utils.batch_render_tn(maps, eval_light)

	if pad_output:
		pad_size = (output_map_size - maps[0].shape[0]) // 2
		render = torch.nn.functional.pad(render, (0, 0, pad_size, pad_size, pad_size, pad_size))
		for idx, map in enumerate(maps):
			maps[idx] = torch.nn.functional.pad(map, (0, 0, pad_size, pad_size, pad_size, pad_size))

	file_utils.write_image_as_png(
		args.output_folder + prefix + "baseColor" + suffix + ".png",
		render_utils.gammaCorrection(
		maps[0].detach()).cpu().numpy(), verbose=False)

	file_utils.write_image_as_png(
		args.output_folder + prefix + "metallic" + suffix + ".png",
		maps[1].detach().cpu().numpy(), verbose=False)

	file_utils.write_normal(
		args.output_folder + prefix + "normal" + suffix + ".png",
		maps[2].detach().cpu().numpy(), opengl_normals=False, verbose=False)

	file_utils.write_image_as_png(
		args.output_folder + prefix + "roughness" + suffix + ".png", torch.pow(
		maps[3], 0.5).detach().cpu().numpy(), verbose=False)

	file_utils.write_image_as_png(
		args.output_folder + prefix + "render" + suffix + ".png",
		render.detach().cpu().numpy(), verbose=False)

	return