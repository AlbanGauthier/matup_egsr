import torch

class FullyConnected(torch.nn.Module):

	non_linearity_dict = {
        "ReLU": 		torch.nn.ReLU(),
        "Sigmoid": 		torch.nn.Sigmoid(),
        "Hardsigmoid": 	torch.nn.Hardsigmoid(),
		"LeakyReLU": 	torch.nn.LeakyReLU()
    }

	def define_mlp(self):

		if self.nb_hidden_layers == 0:
			print("unsupported nb_hidden_layers=0")
			exit(0)
		
		init_conv_1x1 = torch.nn.Conv2d(
				self.num_in, self.hidden_layer_size, 
				kernel_size = 	(self.tile_width, self.tile_width), 
				stride = 		(1, 1))
		
		seq = [init_conv_1x1, self.non_linearity]

		for _ in range(0, self.nb_hidden_layers - 1):
			seq.append(torch.nn.Conv2d(self.hidden_layer_size, self.hidden_layer_size, 1))
			seq.append(self.non_linearity)

		if self.nb_hidden_layers >= 1:
			seq.append(torch.nn.Conv2d(self.hidden_layer_size, self.num_out, 1))
		
		self.mlp = torch.nn.Sequential(*seq)


	def __init__(self, num_in, num_out, tile_width, nb_hidden_layers, hidden_layer_size, non_lin):

		super(FullyConnected, self).__init__()

		self.num_in 			= num_in
		self.num_out 			= num_out
		self.tile_width 		= tile_width
		self.nb_hidden_layers 	= nb_hidden_layers
		self.hidden_layer_size 	= hidden_layer_size
		self.non_linearity 		= self.non_linearity_dict[non_lin]
		
		self.define_mlp()

	def forward(self, x):
		return self.mlp(x)