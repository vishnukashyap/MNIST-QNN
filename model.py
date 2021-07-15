import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.inject.defaults import Int8ActPerTensorFloat

class LinearModel(nn.Module):
	def __init__(self):
		super(LinearModel,self).__init__()

		self.quantize_input =  qnn.QuantIdentity(
			act_quant=Int8ActPerTensorFloat,
			return_quant_tensor=True)

		self.input_layer = nn.Sequential(*[
			qnn.QuantLinear(
				in_features = 784,
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

		output = input_tensor.reshape([-1,784])
		output = self.quantize_input(output)
		output = self.input_layer(output)
		output = self.hidden_layer(output)
		output = self.output_layer(output)

		return output


class LeNet(nn.Module):
	def __init__(self):
		super(LeNet,self).__init__()

		self.quantize_input = qnn.QuantIdentity(
			act_quant=Int8ActPerTensorFloat,
			return_quant_tensor=True)

		self.conv1 = qnn.QuantConv2d(
			in_channels=3,
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

		output = self.quantize_input(input_tensor)
		
		output = self.relu1(self.conv1(output))
		output = F.max_pool2d(output,2)

		output = self.relu2(self.conv2(output))
		output = F.max_pool2d(output,2)

		output = output.reshape(output.shape[0],-1)
		output = self.relu3(self.fc1(output))
		output = self.relu4(self.fc2(output))
		output = self.fc3(output)

		return output