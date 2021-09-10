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

def selective_sampling_outputs(output,target,threshold,loss_criterion):
	'''
		Selectively sample the outputs which would be back-propagated based on the threshold
	'''

	error, _ = torch.topk(abs(output-target),k=1,dim=1)
	indices = error < threshold
	indices = indices.squeeze(1)

	sampled_output =  torch.empty(output.size()).to(output.device.type)

	for i in range(output.size()[0]):
		if indices[i]:
			sampled_output[i] = target[i]
		else:
			sampled_output[i] = output[i]

	loss = loss_criterion(sampled_output,target)

	return loss

def get_classification_accuracy(output,target):
	'''
		This function returns the number of correct predictions and the total number of predictions
	'''
	_ , output_idx = torch.topk(output,k=1,dim=1)
	output_idx = output_idx.squeeze(1)

	correct = (output_idx==target).sum().item()

	total = int(target.shape[0])

	return int(correct),total

def compute_model_accuracy(validation_dataloader,model,model_type,device):
	'''
		This function calculates the validation accuracy and loss of the provided model on the given dataset
	'''
	model.eval()

	loss_criterion = torch.nn.CrossEntropyLoss()

	with torch.no_grad():

		correct = 0.
		total = 0.
		total_loss = 0.
		count = 0

		for batch_idx,(data,target) in enumerate(validation_dataloader):
			data = data.to(device)
			target = target.to(device)

			if model_type ==  "Linear" or model_type == "qnn_Linear":
				data = data.reshape(data.shape[0],-1)

			output,_ = model(data)

			loss = loss_criterion(output,target)
			batch_correct,batch_total = get_classification_accuracy(output,target)

			correct += batch_correct
			total += batch_total
			
			total_loss = (count*total_loss + loss.item())/(count+1)
			count += 1

		accuracy = (correct/total)*100

		return accuracy,total_loss