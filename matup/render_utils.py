import torch

from . import file_utils
from . import renderer

# from nvdiffmodelling
def dot(x: torch.Tensor, y: torch.Tensor, dim_) -> torch.Tensor:
    return torch.sum(x * y, dim=dim_, keepdim=True)

# from nvdiffmodelling
def length(x: torch.Tensor, dim=-1, eps: float = 1e-20) -> torch.Tensor:
	# Clamp to avoid nan gradients because grad(sqrt(0)) = NaN
    return torch.sqrt(torch.clamp(dot(x, x, dim), min=eps))

# from nvdiffmodelling
def safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / length(x, eps)


def compute_normal_from_slopes(normals_xy):
	assert normals_xy.shape[-1] == 2
	norm = torch.clamp(length(normals_xy), min = 0.999)
	normals_xy = 0.999 * normals_xy / norm
	squared_xy = torch.square(normals_xy)
	if normals_xy.ndim == 2:
		z_vec = torch.sqrt(torch.clamp(
			1 - squared_xy[:, 0] - squared_xy[:, 1], 
			min=0.001))
		normal = torch.cat(
			(normals_xy, torch.unsqueeze(z_vec, dim=1)), 
			dim=1)
	else:
		z_vec = torch.sqrt(torch.clamp(
			1 - squared_xy[..., 0:1] - squared_xy[..., 1:2], 
			min=0.001))
		normal = torch.cat((normals_xy, z_vec), dim=-1)
	return normal


def reshape_maps_input_render(map_list):
	maps = []
	for i, map in enumerate(map_list):
		if map is not None:
			maps.append(torch.reshape(map, 
				(map.shape[0]*map.shape[1], map.shape[2])))
	return maps


def batch_render_tn(maps, wi_vec, intensity=None):

	w_o = renderer.Renderer.z_vector

	dims = maps[0].shape[:2]
	maps = reshape_maps_input_render(maps)
	
	# expects T.ndim == 2 (H*W, N)
	render = renderer.Renderer.torch_Render(*maps, wi_vec, w_o, intensity)
	
	render = torch.reshape(render, 
		(dims[0], dims[1], 3, wi_vec.shape[0] * w_o.shape[0])).squeeze()

	render = process_raw_render(render)

	assert not torch.any(torch.isfinite(render) == False)
	assert not torch.any(torch.isnan(render))
	
	return render


def batch_render_tn_explicit(baseColor, metallic, normal, roughness, position3D, wi_vec):

	w_o = renderer.Renderer.z_vector

	dims = baseColor.shape[:2]

	baseColor2 	= torch.reshape(baseColor, 	(baseColor.shape[0] * baseColor.shape[1], 	baseColor.shape[2]))
	metallic2 	= torch.reshape(metallic, 	(metallic.shape[0] 	* metallic.shape[1], 	metallic.shape[2]))
	normal2 	= torch.reshape(normal, 	(normal.shape[0] 	* normal.shape[1], 		normal.shape[2]))
	roughness2 	= torch.reshape(roughness, 	(roughness.shape[0] * roughness.shape[1], 	roughness.shape[2]))
	position3D2 = torch.reshape(position3D, (position3D.shape[0]* position3D.shape[1], 	position3D.shape[2]))
	
	# expects T.ndim == 2 (H*W, N)
	render = renderer.Renderer.torch_Render(
		baseColor2, metallic2, normal2, roughness2, position3D2, wi_vec, w_o, None)
	
	render = torch.reshape(render, 
		(dims[0], dims[1], 3, wi_vec.shape[0] * w_o.shape[0])).squeeze()

	render = process_raw_render(render)
	
	return render


## Building an Orthonormal Basis, Revisited
def branchlessONB(n):
	sign = torch.sign(n[:,2])
	a = -1.0 / (sign + n[:,2])
	b = n[:,0] * n[:,1] * a
	b1 = torch.cat([
		1.0 + sign * n[:,0] * n[:,0] * a, 
		sign * b, -sign * n[:,0]], dim=1)
	b2 = torch.cat([
		b, sign + n[:,1] * n[:,1] * a, 
		-n[:,1]], dim=1)
	return b1, b2


def reinhardTonemapper(t):
	return t / (1 + t)


def neuMIPTonemapper(t):
	return torch.log(t + 1)


def gammaCorrection(input):
	"""linrgb2srgb"""
	limit = 0.0031308
	return torch.where(
		input > limit,
		1.055 * torch.pow(
			torch.clamp(
				input, min=limit), 
				(1.0 / 2.4)) - 0.055,
		12.92 * input)


def DeschaintrelogTensor(in_tensor):
	log_001 = torch.log(0.01)
	div_log = torch.log(1.01)-log_001
	return torch.log(in_tensor.add(0.01)).add(-log_001).div(div_log)


def process_raw_render(render):
	render = reinhardTonemapper(render)
	render = gammaCorrection(render)
	return render