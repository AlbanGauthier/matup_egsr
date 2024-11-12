import torch
import cv2
import numpy as np
import os
import glob

from einops import rearrange
from tqdm import tqdm

from matup.renderer import Renderer

from . import data_utils
from . import file_utils


def load_exr_render_from_folder(folder):
	images = []
	for filename in os.listdir(folder):
		if filename.endswith('.exr'):
			img = read_exr_render(os.path.join(folder, filename))
		if img is not None:
			images.append(img)
	return np.moveaxis(np.array(images), 0, -1), len(images)


def read_exr_render(img_path):
	image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
	assert image is not None
	if image.ndim > 2:
		image = image[:, :, 0]  # keep only x channel
	return image


def load_roughness(images_path, upscale=None):
	roughnessmap = file_utils.read_img_as_float(
		images_path + "/roughness.png",
		multi_channel=False, upscale=upscale)
	# clamp the min roughness values to the min roughness (see renderer.py)
	min_roughness = 0.045
	roughnessmap[roughnessmap < min_roughness] = min_roughness
	return torch.from_numpy(roughnessmap).float()


def load_normal(images_path, opengl_normals, upscale=None):
	normalmap = torch.from_numpy(file_utils.read_img_as_float(
		images_path + "/normal.png",
		multi_channel=True, upscale=upscale))
	return unpack_normal(normalmap, opengl_normals).float()


def load_albedo(images_path, upscale=None):
	img = torch.from_numpy(file_utils.read_img_as_float(
		images_path + "/baseColor.png",
		True, upscale=upscale))  # multi-channel
	return file_utils.srgb2linrgb(img)


def load_metallic(images_path, upscale=None):
	img = file_utils.read_img_as_float(
		images_path + "/metallic.png",
		False, upscale=upscale)  # multi-channel
	return torch.from_numpy(img).float()


def load_height(images_path, upscale=None):
	img = file_utils.read_img_as_float(
		images_path + "/height.png",
		False, upscale=upscale)  # multi-channel
	return torch.from_numpy(img).float()


def unpack_normal(normals, opengl_normals):
	normals = 2.0 * normals - 1.0
	if opengl_normals:
		normals[..., 1] = -normals[..., 1]
	return normals


class MatUpDataset(torch.utils.data.Dataset):

	"""
	A MatUp dataset is organized as follows:

	root/
	+- data/
	|   +- material_name/
	|   |   +- SVBRDF/
	|   |   |   +- baseColor.png
	|   |   |   +- height.png
	|   |   |   +- metallic.png
	|   |   |   +- normal.png
	|   |   |   +- roughness.png
	|	|   +- radiance_LR/
	|   |   |   +- 0000_LR_render.png
	|   |   |   +- 0001_LR_render.png
	|   |   |   +- 0002_LR_render.png
	|   |   |   +- ...
	|	|	+- radiance_SwinIR/
	|   |   |   +- render_0000_SwinIR.png
	|   |   |   +- render_0001_SwinIR.png
	|   |   |   +- render_0002_SwinIR.png
	|   |   |   +- ...
	"""

	def load_material(self, args):
		
		albedo = load_albedo(args.input_mat)
		metallic = load_metallic(args.input_mat)
		normal = load_normal(args.input_mat, self.opengl_normals)
		roughness = load_roughness(args.input_mat)
		height = load_height(args.input_mat)

		# alpha = roughness² is the perceptually linear roughness which is squared later during shading
		alpha = torch.clamp(torch.pow(roughness, 2.0), min=Renderer.min_alpha, max=1.0)
		height = self.disp_val * (height - 0.5)

		input_maps = [albedo, metallic, normal[..., :2], alpha, height]

		upsampled_maps = []
		for map in input_maps:
			map = torch.from_numpy(
				cv2.resize(
        			map.numpy(), 
               		(map.shape[0] * 4, map.shape[1] * 4), 
					interpolation = cv2.INTER_LINEAR))
			if map.ndim == 2:
				map = rearrange(map, 'h w -> h w 1')
			upsampled_maps.append(map)

		self.svbrdf_init = torch.cat(tuple(upsampled_maps[:-1]), dim=-1).float()
		self.height_map = upsampled_maps[-1].squeeze()

	def load_renderings(self, args):
		filename_list = sorted(glob.glob(os.path.join(args.upsamplings_dir, '*')))
		for i, filename in enumerate(filename_list):
			data = file_utils.read_img_as_float(filename, multi_channel=True)
			if self.GT_renders is None:
				self.output_map_size = data.shape[0]
				self.base_res = data.shape[0] // 4
				self.GT_renders = torch.zeros(
					(data.shape[0], data.shape[1], 3, len(filename_list)), 
					dtype=torch.float32)
			self.GT_renders[..., i] = torch.from_numpy(data)
		return 

	def __init__(self, args, device):

		self.svbrdf_init 	= None
		self.height_map		= None
		self.GT_renders		= None
		
		self.output_map_size = 0
		self.base_res 		= 0
		self.disp_val 		= 0.01
		self.opengl_normals = args.opengl_normals
		self.device 		= device

		self.load_material(args)
		self.load_renderings(args)

		# create XY position maps
		self.position2D = data_utils.compute_2D_position_grid(self.output_map_size)
		
		self.tile_width = args.tile_width
		self.half_tw = self.tile_width // 2
		self.out_width = self.output_map_size - self.tile_width + 1
		self.nb_tiles = self.out_width ** 2

	def __getitem__(self, index):

		tile_idx = index % self.nb_tiles

		tile_x = tile_idx % self.out_width
		tile_y = tile_idx // self.out_width

		svbrdf_sample = self.svbrdf_init[
			tile_y: tile_y + self.tile_width,
			tile_x: tile_x + self.tile_width, ...]

		GT_sample = self.GT_renders[
			tile_y + self.half_tw: tile_y + self.half_tw + 1,
			tile_x + self.half_tw: tile_x + self.half_tw + 1]

		position2D_sample = self.position2D[
			tile_y + self.half_tw: tile_y + self.half_tw + 1,
			tile_x + self.half_tw: tile_x + self.half_tw + 1, :]
		
		height_sample = self.height_map[ 		
			tile_y + self.half_tw: tile_y + self.half_tw + 1,
			tile_x + self.half_tw: tile_x + self.half_tw + 1][..., None]
			
		position3D_sample = torch.cat([position2D_sample, height_sample], dim=-1).float()
		
		return svbrdf_sample, GT_sample, position3D_sample


	def __len__(self):
		return self.nb_tiles
	
	def get_svbrdf_init(self):
		return self.svbrdf_init.clone()

	def get_position3D(self):
		return torch.cat([
      		self.position2D, 
        	self.height_map[..., None]], 
                dim=-1).float()