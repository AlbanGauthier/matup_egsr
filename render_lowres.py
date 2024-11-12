import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import torch
import argparse

import matup.data_utils as data_utils
import matup.file_utils as file_utils
import matup.light_utils as light_utils
import matup.render_utils as render_utils
import matup.dataLoader as dataLoader
from matup.renderer import Renderer

def load_maps(args, device):

    baseColor	= dataLoader.load_albedo(args.input_mat, args.AA_upscale)
    metallic 	= dataLoader.load_metallic(args.input_mat, args.AA_upscale).unsqueeze(-1)
    normal 		= dataLoader.load_normal(args.input_mat, args.opengl_normals, args.AA_upscale)
    roughness 	= dataLoader.load_roughness(args.input_mat, args.AA_upscale).unsqueeze(-1)
    height 		= dataLoader.load_height(args.input_mat, args.AA_upscale)

    alpha = torch.clamp(roughness ** 2.0, min=Renderer.min_alpha)
    height = args.disp_val * (height - 0.5)

    upscale = args.AA_upscale
    if args.AA_upscale == None:
        upscale = 1

    size_AA = int(upscale * args.crop_size)
    
    position3D = data_utils.compute_3D_position_grid(size_AA)
    position3D[..., -1] = height[:size_AA, :size_AA]

    maps = [
        baseColor[	:size_AA, :size_AA, :].to(device),
        metallic[	:size_AA, :size_AA, :].to(device),
        normal[		:size_AA, :size_AA, :].to(device),
        alpha[		:size_AA, :size_AA, :].to(device),
        position3D[	:size_AA, :size_AA, :].to(device)
    ]

    return maps
	
def main_render(args):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    Renderer.init_renderer(args, device)

    maps = load_maps(args, device)

    # create the output dir if it doesn't exist
    if not os.path.exists(args.render_dir):
        os.makedirs(args.render_dir)

    wi_vec = 1.4 * light_utils.generate_hemisphere_pts(
        "Fibonacci", sample_count=args.nb_lights,
        near_z=False).float().to(device)

    with torch.no_grad():
        for i in range(wi_vec.shape[0]):

            render = render_utils.batch_render_tn(maps, wi_vec[i:i+1, :])
            
            if args.AA_upscale is not None:
                render = data_utils.average_patches(render, ks=args.AA_upscale)

            file_utils.write_image_as_png(
                os.path.join(args.render_dir, str(i).zfill(4) + "_LR_render.png"), 
                render.detach().cpu().numpy())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_mat", 		type=str, 	default="./data/victorian_wood/SVBRDF/")
    parser.add_argument("--render_dir", 	type=str,	default="./data/victorian_wood/radiance_LR/")

    parser.add_argument('--crop_size',      type=int, 	default=128)
    parser.add_argument('--AA_upscale',     type=int, 	default=2,      help="antialiasing samples, default:2x2")
    parser.add_argument('--nb_lights',      type=int, 	default=100)
    parser.add_argument('--disp_val',       type=float, default=0.01)

    parser.add_argument("--light_falloff",	dest='light_falloff',  action='store_true')
    parser.add_argument("--ogl_normals",	dest='opengl_normals', action='store_true', 
                        help='input openGL-style normal maps instead of DirectX')

    parser.set_defaults(opengl_normals		= False)
    parser.set_defaults(light_falloff		= False)

    args = parser.parse_args()
    main_render(args)
