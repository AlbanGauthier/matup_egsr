import torch
import cv2
import json, os, time, re
import numpy as np
import matplotlib.pyplot as plt
from subprocess import PIPE, run
from argparse import Namespace

from . import data_utils
from . import render_utils
from . import renderer

def out(command):
    result = run(
		command, stdout=PIPE, stderr=PIPE, 
		universal_newlines=True, shell=True)
    return result.stdout


def exit_with_message(message):
	print("##############################################")
	print(message)
	print("##############################################")
	exit(0)

def get_training_id(args):
	t = time.localtime()
	current_time = time.strftime("%b%d_%Hh%M", t)
	res_str = ""
	res_str += current_time
	res_str += "_lvl" + str(args.nb_levels)
	res_str += "_bs" + str(args.batch_size)
	res_str += "_" + str(args.hid_nb_A) + "-" + str(args.hid_size_A)
	res_str += "_" + str(args.hid_nb_B) + "-" + str(args.hid_size_B)
	res_str += "_" + str(args.learning_rate)
	res_str += "/"
	return res_str

def get_training_time():
	t = time.localtime()
	res_str = time.strftime("%b%d_%Hh%Mm%Ss", t)
	return res_str

def get_dict_from_model(model):
	# works only for model defined with self.func = Sequential(...)
	modules = model.named_modules()
	ret = ""
	for m in modules:
		if m[0] == 'mlp':
			ret = repr(m[1])
			return ret
	return "__error__ in get_dict_from_model()"

def process_image_as_float(img, multi_channel=True):
    img_type = img.dtype
    if multi_channel:  # converting BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if (img_type == np.dtype('uint16')):
        img = img / float(pow(2, 16) - 1)
    elif (img_type == np.dtype('uint8')):
        img = img / float(pow(2, 8) - 1)
    return img

def read_img_as_float(img_path, multi_channel=True, upscale=None):
	image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
	if upscale is not None:
		H = image.shape[0]
		W = image.shape[1]
		s = upscale
		image = cv2.copyMakeBorder(image, 2, 2, 2, 2,
			borderType=cv2.BORDER_WRAP)
		image = cv2.resize(image, (int((H+4)*s), int((W+4)*s)),
			interpolation = cv2.INTER_LINEAR)
		if image.ndim == 2:
			image = image[int(2*s):int(s*(H+2)), int(2*s):int(s*(W+2))]
		elif image.ndim == 3:
			image = image[int(2*s):int(s*(H+2)), int(2*s):int(s*(W+2)), :]
		else:
			exit_with_message("error in read_img_as_float")
	if image is None:
		exit_with_message(
			"No file at specified path: " + img_path)
	# roughness sometimes has 3 channels
	if not multi_channel and image.ndim > 2:
		image = image[:, :, 0]
	float_img = process_image_as_float(image, multi_channel)
	if img_path.endswith("normal.png"):
		float_img = data_utils.normalize_np_norm(float_img)
	return float_img

def srgb2linrgb(input_color):
    limit = 0.04045
    transformed_color = torch.where(
        input_color > limit,
        torch.pow((torch.clamp(input_color, min=limit) + 0.055) / 1.055, 2.4),
        input_color / 12.92
    )  # clamp to stabilize training
    return transformed_color

def write_normal(path, array, opengl_normals, verbose=False):
	if opengl_normals:
		array[..., 1] = -array[..., 1]
	array = np.array(65535 * (0.5 * array + 0.5), dtype=np.uint16)
	array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
	if not cv2.imwrite(path, array):
		print("error saving normal map at:", path)
		assert False
	if verbose:
		print("saved normal map: " + path)


def write_image_as_png(path, vec, verbose=False):
	assert path.endswith(".png")
	array = np.clip(vec, 0.0, 1.0)
	array = np.array(65535 * array, dtype=np.uint16)
	if array.ndim==3:  # converting BGR to RGB
		if array.shape[2] == 4:
			array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)
		elif array.shape[2] == 3:
			array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
	if not cv2.imwrite(path, array):
		print("error saving img at:", path)
		assert False
	if verbose:
		print("saved image in: " + path)

def write_image_as_exr(path, vec):
	if path.endswith(".png"):
		path = path[:-4]
	if not path.endswith(".exr"):
		path += ".exr"
	if vec.ndim==3:
		# pad with one channel of zeros
		if vec.shape[2] == 2:
			vec = np.pad(vec, ((0, 0), (0, 0), (0, 1)))
		# converting RGB to BGR
		vec = cv2.cvtColor(vec, cv2.COLOR_RGB2BGR)
	if cv2.imwrite(path, vec):
		print("saved image in: " + path)
	else:
		print("error saving img")
		assert False