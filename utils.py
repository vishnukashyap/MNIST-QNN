import torch

def get_layer_parameters(model,idx_range):
	'''
		get the learnable weights for the layer being trained
	'''
	param_out = []
	param_out_name = []
	for it, (name, param) in enumerate(model.named_parameters()):
		if it in idx_range:
			param_out.append(param)
			param_out_name.append(name)
	return param_out, param_out_name

def to_categorical(y,num_classes):
	'''
		1 hot encoding of the targets
	'''
	return torch.squeeze(torch.eye(num_classes)[y])

def selective_sampling_outputs(output,target,threshold):
	'''
		Selectively sample the outputs which would be back-propagated based on the threshold
	'''

	error, _ = torch.topk(abs(output-target),k=1,dim=1)
	indices = error < threshold
	indices = indices.squeeze(1)

	output[indices] = target[indices]

	return output