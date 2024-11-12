import os
import sys
import argparse

from render_lowres import main_render
from main_optim import main_optim

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("--upscaler", 		type=str, 	default="swinir", choices=["swinir", "srflow"]) 

	#####
	# I/O args
	#####
	parser.add_argument("--input_mat", 		type=str, 	default="./data/victorian_wood/SVBRDF/") 
	parser.add_argument("--render_dir", 	type=str,	default="./data/victorian_wood/radiance_LR/")
	parser.add_argument("--upsamplings_dir",type=str,	default="./data/victorian_wood/radiance_Upscaled/")
	parser.add_argument("--output_folder", 	type=str,	default="./output/")
	parser.add_argument("--output_init",	dest='output_init', action='store_true', help="output initial bilinear upsampling of maps")

	#####
	# Optimization args
	#####
	parser.add_argument("--tile_width", 	type=int, 	default=17, 	choices=[5, 9, 13, 17, 25])
	parser.add_argument("--batch_size", 	type=int, 	default=128, 	choices=[1, 4, 16, 64, 128, 256])
	parser.add_argument('--nb_epochs', 		type=int, 	default=10)
	parser.add_argument('--learning_rate', 	type=float, default=1e-4)
	parser.add_argument('--alpha_reg', 		type=float, default=1e-1)
	parser.add_argument("--save_interval",	type=int, 	default=999)

	parser.add_argument("--hid_layer_width",type=int, 	default=128)
	parser.add_argument("--hid_layer_size",	type=int, 	default=4)
	parser.add_argument("--non_linearity", 	type=str, 	default="LeakyReLU", choices=["ReLU", "Sigmoid", "Hardsigmoid", "LeakyReLU"])

	#####
	# Render args
	#####
	parser.add_argument('--crop_size',      type=int, 	default=128)
	parser.add_argument('--AA_upscale',     type=int, 	default=2,      help="antialiasing samples, default:2x2")
	parser.add_argument('--nb_lights',      type=int, 	default=100)
	parser.add_argument('--disp_val',       type=float, default=0.01)
	parser.add_argument("--light_falloff",	dest='light_falloff',  action='store_true')
	parser.add_argument("--ogl_normals",	dest='opengl_normals', action='store_true', 
						help='input openGL-style normal maps instead of DirectX')

	#####
	# Upsample args
	#####
	parser.add_argument('--temperature',    type=float, default=0.5)
	parser.add_argument('--material_size',  type=int, 	default=128)
	parser.add_argument("--up_batch_size",	type=int, 	default=10,     help="nb of parallel upsamplings, must divide nb_renders")

	parser.set_defaults(output_init			= False)
	parser.set_defaults(light_falloff		= False)
	parser.set_defaults(opengl_normals		= False)

	args = parser.parse_args()

	#####
	# Render lowres material
	#####
	main_render(args)

	#####
	# Upsample lowres material
	#####
	if "swinir" in args.upscaler:
		from sr_swinir import main_upsample_swinir
		main_upsample_swinir(args)
	else: # "SRFlow"
		from sr_srflow import main_upsample_srflow
		main_upsample_srflow(args)

	#####
	# Optimize highres material
	#####
	main_optim(args)

	print("...done")