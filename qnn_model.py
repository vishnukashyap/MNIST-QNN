import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.inject.defaults import Int8ActPerTensorFloat

dataset_input_channels_linear = {"MNIST":784,"CIFAR10":3072}
dataset_input_channels_conv = {"MNIST":1,"CIFAR10":3}

'''
	This file contains quantized model definitions 
'''

class LinearModel(nn.Module):
	def __init__(self,dataset_name):
		super(LinearModel,self).__init__()

		input_channels = dataset_input_channels_linear[dataset_name]

		self.quantize_input =  qnn.QuantIdentity(
			act_quant=Int8ActPerTensorFloat,
			return_quant_tensor=True)

		self.input_layer = nn.Sequential(*[
			qnn.QuantLinear(
				in_features = input_channels,
				out_features = 128,
				bias=True,
				weight_quant_type=QuantType.BINARY,
				return_quant_tensor=True),
			qnn.QuantReLU(
				input_quant=Int8ActPerTensorFloat,
				act_quant=Int8ActPerTensorFloat,
				return_quant_tensor=True)])

		self.hidden_layer = nn.Sequential(*[
			qnn.QuantLinear(
				in_features = 128,
				out_features = 128,
				bias=True,
				weight_quant_type=QuantType.BINARY,
				return_quant_tensor=True),
			qnn.QuantReLU(
				input_quant=Int8ActPerTensorFloat,
				act_quant=Int8ActPerTensorFloat, 
				return_quant_tensor=True)])

		self.output_layer = nn.Sequential(*[
			qnn.QuantLinear(
				in_features = 128,
				out_features = 10,
				bias=True,
				weight_quant_type=QuantType.BINARY,
				return_quant_tensor=True),
			qnn.QuantSigmoid(
				input_quant=Int8ActPerTensorFloat,
				act_quant=Int8ActPerTensorFloat, 
				return_quant_tensor=True)])

	def forward(self,input_tensor):

		hidden_values = []

		output = self.quantize_input(output)
		hidden_values.append(output)
		
		output = self.input_layer(output)
		hidden_values.append(output)
		
		output = self.hidden_layer(output)
		hidden_values.append(output)

		output = self.output_layer(output)
		hidden_values.append(output)

		return output,hidden_values


class LeNet(nn.Module):
	def __init__(self,dataset_name):
		super(LeNet,self).__init__()

		input_channels = dataset_input_channels_conv[dataset_name]

		self.quantize_input = qnn.QuantIdentity(
			act_quant=Int8ActPerTensorFloat,
			return_quant_tensor=True)

		self.conv1 = qnn.QuantConv2d(
			in_channels=input_channels,
			out_channels=6,
			kernel_size=5,
			weight_quant=QuantType.BINARY,
			return_quant_tensor=True)

		self.relu1 = qnn.QuantReLU(
			input_quant=Int8ActPerTensorFloat,
			act_quant=Int8ActPerTensorFloat, 
			return_quant_tensor=True)

		self.conv2 = qnn.QuantConv2d(
			in_channels=6,
			out_channels=16,
			kernel_size=5,
			weight_quant=QuantType.BINARY,
			return_quant_tensor=True)

		self.relu2 = qnn.QuantReLU(
			input_quant=Int8ActPerTensorFloat,
			act_quant=Int8ActPerTensorFloat, 
			return_quant_tensor=True)

		self.fc1 = qnn.QuantLinear(
			in_features=16*5*5,
			out_features=120,
			bias=True,
			weight_quant_type=QuantType.BINARY,
			return_quant_tensor=True)

		self.relu3 = qnn.QuantReLU(
			input_quant=Int8ActPerTensorFloat,
			act_quant=Int8ActPerTensorFloat,
			return_quant_tensor=True)

		self.fc2 = qnn.QuantLinear(
			in_features=120,
			out_features=84,
			bias=True,
			weight_quant_type=QuantType.BINARY,
			return_quant_tensor=True)

		self.relu4 = qnn.QuantReLU(
			input_quant=Int8ActPerTensorFloat,
			act_quant=Int8ActPerTensorFloat,
			return_quant_tensor=True)

		self.fc3 = qnn.QuantLinear(
			in_features=84,
			out_features=10,
			bias=True,
			weight_quant_type=QuantType.BINARY,
			return_quant_tensor=True)

	def forward(self,input_tensor):

		hidden_values = []

		output = self.quantize_input(input_tensor)
		
		output = self.relu1(self.conv1(output))
		output = F.max_pool2d(output,2)
		hidden_values.append(output)

		output = self.relu2(self.conv2(output))
		output = F.max_pool2d(output,2)
		hidden_values.append(output)

		output = output.reshape(output.shape[0],-1)
		output = self.relu3(self.fc1(output))
		hidden_values.append(output)

		output = self.relu4(self.fc2(output))
		hidden_values.append(output)

		output = self.fc3(output)
		hidden_values.append(output)

		return output,hidden_values