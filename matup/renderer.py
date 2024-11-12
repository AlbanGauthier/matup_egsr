import torch
import numpy as np
import torch.nn as nn

from . import render_utils
from . import data_utils
from . import file_utils

class Renderer:

	z_vector    = torch.from_numpy(np.array([0,0,1])).float()
	white_light = torch.from_numpy(np.array([1,1,1])).float()

	default_light_intensity = 8.0
	min_roughness           = 0.045
	min_alpha           	= 2e-3
	min_alpha_sqr			= 4e-6
	min_alpha_4				= 1.6e-11
	min_visibility         	= 1e-8
	min_ndf_denom         	= 1e-8
	oneOverPi               = 1 / np.pi

	render_device = ''
	render_resolution       = -1

	light_falloff	= False
	is_init         = False
			
	@staticmethod
	def init_renderer(args, device):
		Renderer.z_vector = torch.reshape(
			Renderer.z_vector, (1, 3)).to(device)
		Renderer.white_light = torch.reshape(
			Renderer.white_light, (1, 3, 1)).to(device)
		Renderer.render_device = device
		Renderer.light_falloff = args.light_falloff
		Renderer.is_init = True

	@staticmethod
	def set_render_resolution(res):
		Renderer.render_resolution = res

	@staticmethod
	def get_render_resolution():
		return Renderer.render_resolution

	@staticmethod
	def D_GGX_Anisotropic(a_sqr, b_sqr, c_sqr, wh, T, B, NdotH):
		ToH = torch.sum(wh * T, dim=1, keepdim=True)
		BoH = torch.sum(wh * B, dim=1, keepdim=True)
		det_2 = torch.clamp(a_sqr*b_sqr - c_sqr*c_sqr, min=1e-15)
		denom = torch.sqrt(det_2) * torch.pow(
			(a_sqr*ToH*ToH / det_2 
			+ 2*c_sqr*ToH*BoH / det_2 
			+ b_sqr*BoH*BoH / det_2)
			+ NdotH*NdotH, 2.0)
		denom = torch.clamp(denom, min=Renderer.min_ndf_denom)
		return Renderer.oneOverPi / denom

	@staticmethod
	def V_SmithGGXCorrelated_aniso(
		a_sqr, b_sqr, c_sqr, T, B, NoL, NoV, L, V):

		ToV = torch.sum(V * T, dim=1, keepdim=True)
		ToL = torch.sum(L * T, dim=1, keepdim=True)
		BoV = torch.sum(V * B, dim=1, keepdim=True)
		BoL = torch.sum(L * B, dim=1, keepdim=True)
		
		lambdaV_V = torch.clamp(
			b_sqr*ToV*ToV + 2*c_sqr*ToV*BoV + a_sqr*BoV*BoV + NoV*NoV, min=1e-15)
		lambdaV_L = torch.clamp(
			b_sqr*ToL*ToL + 2*c_sqr*ToL*BoL + a_sqr*BoL*BoL + NoL*NoL, min=1e-15)

		lambdaV = NoL.unsqueeze(-1) * torch.sqrt(lambdaV_V).unsqueeze(-2)
		lambdaL = NoV.unsqueeze(-2) * torch.sqrt(lambdaV_L).unsqueeze(-1)

		N, C, I, O = lambdaV.shape
		lambdaV = lambdaV.reshape((N, C, I * O))
		lambdaL = lambdaL.reshape((N, C, I * O))

		lambdaV = torch.clamp(lambdaV, min=Renderer.min_visibility)
		lambdaL = torch.clamp(lambdaL, min=Renderer.min_visibility)
		v = 0.5 / (lambdaV + lambdaL)

		return v

	@staticmethod
	def F_Schlick(u, F0):
		return F0 + (1.0 - F0) * torch.pow(1.0 - u, 5.0)

	@staticmethod
	def torch_render_base(baseColor, metallic, position3D, wi, light_intensity):
		
		assert Renderer.is_init

		wi = wi.t().unsqueeze(0)
		pointToLight = wi - position3D.unsqueeze(-1)
		## optionally add falloff
		if Renderer.light_falloff:
			distance = torch.linalg.norm(pointToLight, dim=1, keepdim=True)
			light_intensity = light_intensity / torch.sqrt(distance)
		wi_out = data_utils.torch_norm(pointToLight, dimToNorm=1)

		baseColor = torch.clamp(baseColor, min=0.0, max=1.0)
		metallic = torch.clamp(metallic, min=0.0, max=1.0)
		
		diffuseColor = (1.0 - metallic) * baseColor
		F0 = 0.04
		F0 = (1 - metallic) * F0 + metallic * baseColor
		F0 = F0.unsqueeze(-1)

		return diffuseColor, F0, wi_out, light_intensity

	@staticmethod
	def torch_GGX_aniso_BRDF(
		normal, a_sqr, b_sqr, c_sqr, 
		diffuseColor, F0, wiNorm, woNorm):

		whNorm = wiNorm.unsqueeze(-1) + woNorm.unsqueeze(-2)
		N, C, I, O = whNorm.shape
		whNorm = whNorm.reshape((N, C, I * O))
		whNorm = data_utils.torch_norm(whNorm, dimToNorm=1)

		NdotH = torch.sum(normal * whNorm, dim=1, keepdim=True)
		NdotV = torch.sum(normal * woNorm, dim=1, keepdim=True)
		NdotL = torch.sum(normal * wiNorm, dim=1, keepdim=True)
		LdotH = torch.sum(wiNorm.repeat(1, 1, O) * whNorm, dim=1, keepdim=True)

		Heavyside_D = torch.where(NdotH <= 0, 0, 1)
		Heavyside_G = torch.where(LdotH <= 0, 0, 1)
		
		NdotH = torch.clamp(NdotH, min=1e-08, max=0.9999)
		NdotV = torch.clamp(NdotV, min=1e-08, max=0.9999)
		NdotL = torch.clamp(NdotL, min=1e-08, max=0.9999)
		LdotH = torch.clamp(LdotH, min=1e-08, max=0.9999)

		tangent, bitangent = render_utils.branchlessONB(normal)
		tangent = tangent.unsqueeze(-1)
		bitangent = bitangent.unsqueeze(-1)

		brdf_tmp = diffuseColor.unsqueeze(-1) * Renderer.oneOverPi

		a_sqr = a_sqr.unsqueeze(-1)
		b_sqr = b_sqr.unsqueeze(-1)
		c_sqr = c_sqr.unsqueeze(-1)

		V = Heavyside_G * Renderer.V_SmithGGXCorrelated_aniso(
			a_sqr, b_sqr, c_sqr, tangent, bitangent,
			NdotL, NdotV, wiNorm, woNorm)

		D = Heavyside_D * Renderer.D_GGX_Anisotropic(
			a_sqr, b_sqr, c_sqr,
			whNorm, tangent, bitangent, NdotH)
		
		F = Renderer.F_Schlick(LdotH, F0)
		specu_tmp = V * F * D

		brdf_tmp = NdotL.repeat(1, 1, O) * (brdf_tmp + specu_tmp)

		return brdf_tmp

	@staticmethod
	def torch_Render_aniso(baseColor, metallic, normal, roughness, 
		position3D, wi, wo, light_intensity):

		diffuseColor, F0, wiNorm, light_intensity = Renderer.torch_render_base(
				baseColor, metallic, position3D, wi, light_intensity)

		woNorm = data_utils.torch_norm(wo, dimToNorm=1).t().unsqueeze(0)
		normal = data_utils.torch_norm(normal, dimToNorm=1).unsqueeze(-1)
		
		a_sqr = torch.pow(roughness, 2.0)
		b_sqr = torch.pow(roughness, 2.0)
		c_sqr = 0.0 * roughness

		renders = Renderer.torch_GGX_aniso_BRDF(
			normal, a_sqr, b_sqr, c_sqr,
			diffuseColor, F0, wiNorm, woNorm)

		renders = light_intensity * renders

		return renders

	@staticmethod
	def torch_Render(albedo, metallic, normal, roughness, 
		position3D, wi, wo, light_intensity = None):

		if light_intensity is None:
			light_intensity = Renderer.default_light_intensity

		return Renderer.torch_Render_aniso(
			albedo, metallic, normal, roughness, 
			position3D, wi, wo, light_intensity)