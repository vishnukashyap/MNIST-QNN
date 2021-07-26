import torch

dataset_input_channels_linear = {"MNIST":784,"CIFAR10":3072}
dataset_input_channels_conv = {"MNIST":1,"CIFAR10":3}

'''
	This file contains model definitions for the regular models
'''

class LinearModel(torch.nn.Module):
	def __init__(self,dataset_name):
		super(LinearModel,self).__init__()

		input_channels = dataset_input_channels_linear[dataset_name]

		self.number_of_hidden_units = 3

		self.input_layer = torch.nn.Sequential(*[
			torch.nn.Linear(
				in_features=input_channels,
				out_features=128,
				bias=True),
			torch.nn.BatchNorm1d(num_features=128),
			torch.nn.ReLU()])

		self.hidden_layer = torch.nn.Sequential(*[
			torch.nn.Linear(
				in_features=128,
				out_features=128,
				bias=True),
			torch.nn.BatchNorm1d(num_features=128),
			torch.nn.ReLU()])

		self.output_layer = torch.nn.Sequential(*[
			torch.nn.Linear(
				in_features=128,
				out_features=10,
				bias=True),
			torch.nn.BatchNorm1d(num_features=10),
			torch.nn.Sigmoid()])

	def forward(self,input_tensor):

		hidden_values = []

		output = self.input_layer(input_tensor)
		hidden_values.append(output)

		output = self.hidden_layer(output)
		hidden_values.append(output)

		output = self.output_layer(output)
		hidden_values.append(output)

		return output,hidden_values

class LeNet(torch.nn.Module):
	def __init__(self,dataset_name):
		super(LeNet,self).__init__()

		input_channels = dataset_input_channels_conv[dataset_name]

		self.number_of_hidden_units = 5

		self.Conv1 = torch.nn.Sequential(*[
			torch.nn.Conv2d(
				in_channels=input_channels,
				out_channels=6,
				kernel_size=5),
			torch.nn.BatchNorm2d(num_features=6),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(2,stride=2)])

		self.Conv2 = torch.nn.Sequential(*[
			torch.nn.Conv2d(
				in_channels=6,
				out_channels=16,
				kernel_size=5),
			torch.nn.BatchNorm2d(num_features=16),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(2,stride=2)])

		self.fully_connected_1 = torch.nn.Sequential(*[
			torch.nn.Linear(
				in_features=16*5*5,
				out_features=120,
				bias=True),
			torch.nn.BatchNorm1d(num_features=120),
			torch.nn.ReLU()])

		self.fully_connected_2 = torch.nn.Sequential(*[
			torch.nn.Linear(
				in_features=120,
				out_features=84,
				bias=True),
			torch.nn.BatchNorm1d(num_features=84),
			torch.nn.ReLU()])

		self.fully_connected_3 = torch.nn.Sequential(*[
			torch.nn.Linear(
				in_features=84,
				out_features=10,
				bias=True),
			torch.nn.BatchNorm1d(num_features=10),
			torch.nn.Sigmoid()])

	def forward(self,input_tensor):

		hidden_values = []

		output = self.Conv1(input_tensor)
		hidden_values.append(output)

		output = self.Conv2(output)
		hidden_values.append(output)

		output = output.reshape(output.shape[0],-1)
		output = self.fully_connected_1(output)
		hidden_values.append(output)

		output = self.fully_connected_2(output)
		hidden_values.append(output)

		output = self.fully_connected_3(output)
		hidden_values.append(output)

		return output, hidden_values
