import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import torch
import shutil
import argparse
import numpy as np

import matup.train as train
import matup.data_utils as data_utils
import matup.file_utils as file_utils
import matup.light_utils as light_utils
from matup.model import FullyConnected
from matup.renderer import Renderer
from matup.dataLoader import MatUpDataset

device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def output_initial_maps(args, dataset):

	padding = (args.tile_width - 1) // 2
	svbrdf_init = dataset.get_svbrdf_init()[
		padding:-padding, padding:-padding, ...].to(device)
	position3D = dataset.get_position3D()[
		padding:-padding, padding:-padding][..., None].to(device)
	maps_init = data_utils.maps_from_data(svbrdf_init, position3D)
	data_utils.output_maps_from_list(
		maps_init, args, suffix="_init", 
		pad_output=True)
	
def main_optim(args):

	np.random.seed(1996)
	torch.manual_seed(1996)

	print('Using device:', device)

	if device.type == 'cuda':
		print(torch.cuda.get_device_name(0))
		print("")

	#Init renderer values (convert to Tensor, send to GPU)
	Renderer.init_renderer(args, device)
		
	current_time = file_utils.get_training_time()

	args.output_folder += current_time
	args.output_folder += "/"
    
	os.makedirs(args.output_folder, exist_ok=True)
	
	######
	# Create model
	######
	model = FullyConnected(
		num_in		= 7, # basecolor, metallic, normalXY, roughness
		num_out		= 7,
		tile_width 			= args.tile_width,
		nb_hidden_layers	= args.hid_layer_size,
		hidden_layer_size	= args.hid_layer_width,
		non_lin				= args.non_linearity
		).float().to(device)
	
	######
	# Optimizer and Losses
	######

	optim_params = [param for param in model.parameters()]
	optimizer = torch.optim.Adam(optim_params, lr=args.learning_rate, capturable=True)
	loss_MSE = torch.nn.MSELoss().to(device)	

	######
	# Create dataset
	######
	matup_dataset = MatUpDataset(args, device)

	data_loader = torch.utils.data.DataLoader(
		matup_dataset, batch_size=args.batch_size,
		num_workers=8, persistent_workers=True,
		shuffle=True, pin_memory=True)
	
	# should be similar to render_lowres.py
	wi_tab = 1.4 * light_utils.generate_hemisphere_pts(
		"Fibonacci", sample_count=args.nb_lights, 
		near_z=False).float().to(device)
	
	if args.output_init:
		output_initial_maps(args, matup_dataset)

	##############
	## Training ##
	##############

	maps_train = train.train_model(args, model, optimizer, matup_dataset, data_loader, loss_MSE, wi_tab, device)

	data_utils.output_maps_from_list(maps_train, args, "_opt", "", pad_output=True)
 
	# remove LR & HR renderings 
	shutil.rmtree(args.render_dir)
	shutil.rmtree(args.upsamplings_dir)

	return



if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--input_mat", 		type=str, 	default="./data/victorian_wood/SVBRDF") 
	parser.add_argument("--upsamplings_dir",type=str, 	default="./data/victorian_wood/radiance_Upscaled") 
	parser.add_argument("--output_folder", 	type=str,	default="./output/")
	
	parser.add_argument("--tile_width", 	type=int, 	default=17, 	choices=[5, 9, 13, 17, 25])
	parser.add_argument("--batch_size", 	type=int, 	default=128, 	choices=[1, 4, 16, 64, 128, 256])
	parser.add_argument('--nb_epochs', 		type=int, 	default=10)
	parser.add_argument('--learning_rate', 	type=float, default=1e-4)
	parser.add_argument('--alpha_reg', 		type=float, default=1e-1)
	parser.add_argument('--nb_lights', 		type=int, 	default=100)
	parser.add_argument("--save_interval",	type=int, 	default=999)

	parser.add_argument("--hid_layer_width",type=int, 	default=128)
	parser.add_argument("--hid_layer_size",	type=int, 	default=4)
	parser.add_argument("--non_linearity", 	type=str, 	default="LeakyReLU", choices=["ReLU", "Sigmoid", "Hardsigmoid", "LeakyReLU"])
	parser.add_argument("--ogl_normals",	dest='opengl_normals', action='store_true', 
					 	help='input openGL-style normal maps instead of DirectX')
	
	parser.set_defaults(opengl_normals		= False)
	
	args = parser.parse_args()
	main_optim(args)
