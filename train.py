import os
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

import qnn_model
import model
import dataset
import utils
import math_funcs

def train(dataset_type,dataset_dir,checkpoint_dir,model_type,batch_size,device,epochs,last_checkpoint,tensorboard_writer,lr,momentum):
	''' Standard training with back propagation'''

	# Initialize and load the dataloader
	if dataset_type == "MNIST":
		train_dataloader,val_dataloader,test_datalaoder = dataset.load_mnist(dataset_dir,batch_size)
	elif dataset_type == "CIFAR10":
		train_dataloader,val_dataloader,test_datalaoder = dataset.load_cifar10(dataset_dir,batch_size)
	else:
		print("Dataset type not supported")
		return

	# Initialize the model
	if model_type == "qnn_Linear":
		dnn_model = qnn_model.LinearModel(dataset_type).to(device)
	elif model_type == "qnn_LeNet":
		dnn_model = qnn_model.LeNet(dataset_type).to(device)
	elif model_type == "Linear":
		dnn_model = model.LinearModel(dataset_type).to(device)
	elif model_type == "LeNet":
		dnn_model = model.LeNet(dataset_type).to(device)
	else:
		print("Model type not supported")
		return

	# Initialize the optimizer and loss criterion
	optimizer = torch.optim.SGD(dnn_model.parameters(),lr=lr,momentum=momentum)
	loss_criterion =  torch.nn.CrossEntropyLoss().to(device)

	cur_epoch = 0
	train_step = 0
	val_step = 0

	train_accuracy = 0.
	train_loss = 0.
	test_loss = 0.
	validation_loss = 0.
	validation_accuracy = 0.

	# Load a previous checkpoint to resume training
	if last_checkpoint != None:
		dnn_model.load_state_dict(last_checkpoint["model"])
		optimizer.load_state_dict(last_checkpoint["optimizer"])
		cur_epoch = last_checkpoint["epoch"]
		train_loss = last_checkpoint["train_loss"]
		validation_loss = last_checkpoint["validation_loss"]
		test_loss = last_checkpoint["test_loss"]
		train_step = last_checkpoint["tensorboard_train_step"]
		val_step = last_checkpoint["tensorboard_val_step"]

		#Initialize the new lr to optimizer
		for groups in optimizer.param_groups:
			groups["lr"] = lr

	# Start Training
	for epoch in range(cur_epoch,epochs):

		progress_bar = tqdm.tqdm(enumerate(train_dataloader))
		dnn_model.train()
		loss_criterion.train()
		progress_bar.set_description("Epoch: "+str(epoch)+" Avg loss of entire dataset: "+str(train_loss))

		for batch_idx,(data,target) in progress_bar:
			data = data.to(device)
			target = target.to(device)
			if model_type == "Linear" or model_type == "qnn_Linear":
				data = data.reshape(-1,np.prod(data.shape[1:]))

			output, _ = dnn_model(data)
			loss = loss_criterion(output,target)
			loss.backward()
			progress_bar.set_description("Epoch: "+str(epoch)+" Batch index: "+str(batch_idx)+" Batch-wise Training Loss: "+str(loss.item()))

			optimizer.step()
			optimizer.zero_grad()

			train_loss = (train_loss*(batch_idx) + loss.item())/(batch_idx+1)
			tensorboard_writer.add_scalar("Batch-wise Training Loss",(loss.item()),global_step=train_step)
			train_step += 1
			del data
			del target
			del output
			del loss
			torch.cuda.empty_cache()

		train_accuracy,_ = utils.compute_model_accuracy(train_dataloader,dnn_model,model_type,device)

		validation_accuracy,validation_loss = utils.compute_model_accuracy(val_dataloader,dnn_model,model_type,device)
		tensorboard_writer.add_scalar("Validation loss",(validation_loss),global_step=val_step)
		val_step += 1

		print("\nEpoch:               "+str(epoch))
		print("Training Loss:       "+str(train_loss))
		print("Training Accuracy:   "+str(train_accuracy))
		print("Validation Loss:     "+str(validation_loss))
		print("Validation Accuracy: "+str(validation_accuracy)+"\n")

		# Save the weights at the end of the each epoch
		checkpoint_file = "Standard_" + model_type+"_"+dataset_type+"_"+"_Checkpoint_"+str(epoch)+".pt"
		
		checkpoint = {}
		checkpoint["model"] = dnn_model.state_dict()
		checkpoint["optimizer"] = optimizer.state_dict()
		checkpoint["train_loss"] = train_loss
		checkpoint["validation_loss"] = validation_loss
		checkpoint["test_loss"] = test_loss
		checkpoint["tensorboard_train_step"] = train_step
		checkpoint["tensorboard_val_step"] = val_step
		save_dir = os.path.join(checkpoint_dir,checkpoint_file)
		torch.save(checkpoint,save_dir)

		del checkpoint
		torch.cuda.empty_cache()

def hsic_train(dataset_type,dataset_dir,checkpoint_dir,model_type,batch_size,device,epochs,last_checkpoint,tensorboard_writer,lr,momentum):
	''' HSIC Bottleneck method of training '''

	# Initialize and load the dataloader
	if dataset_type == "MNIST":
		train_dataloader,val_dataloader,test_datalaoder = dataset.load_mnist(dataset_dir,batch_size)
	elif dataset_type == "CIFAR10":
		train_dataloader,val_dataloader,test_datalaoder = dataset.load_cifar10(dataset_dir,batch_size)
	else:
		print("Dataset type not supported")
		return

	# Initialize the model
	if model_type == "qnn_Linear":
		dnn_model = qnn_model.LinearModel(dataset_type).to(device)
	elif model_type == "qnn_LeNet":
		dnn_model = qnn_model.LeNet(dataset_type).to(device)
	elif model_type == "Linear":
		dnn_model = model.LinearModel(dataset_type).to(device)
	elif model_type == "LeNet":
		dnn_model = model.LeNet(dataset_type).to(device)
	else:
		print("Model type not supported")
		return

	# Initialize the optimizer
	idx_range = []
	it = 0

	for i in range(dnn_model.number_of_hidden_units):
		idx_range.append(np.arange(it,it+2).tolist())
		it += 2
		params,named_params = utils.get_layer_parameters(model=model,idx_range=idx_range[i])
		exec('optimizer_'+str(i) + ' = torch.optim.SGD(params,lr=learning_rate,momentum=0.9)')

	cur_epoch = 0
	train_step = 0
	val_step = 0

	train_accuracy = 0.
	train_loss = 0.
	test_loss = 0.
	validation_loss = 0.
	validation_accuracy = 0.

	# HSIC Variables
	beta = 7.0
	sigma = None

	# Load a previous checkpoint to resume training
	if last_checkpoint:
		dnn_model.load_state_dict(last_checkpoint["model"])
		cur_epoch = last_checkpoint["epoch"]
		train_loss = last_checkpoint["train_loss"]
		validation_loss = last_checkpoint["validation_loss"]
		test_loss = last_checkpoint["test_loss"]
		train_step = last_checkpoint["tensorboard_train_step"]
		val_step = last_checkpoint["tensorboard_val_step"]

	# Start training code
	for epoch in range(cur_epoch,epochs):

		progress_bar = tqdm.tqdm(enumerate(train_dataloader))
		dnn_model.train()
		loss_criterion.train()
		progress_bar.set_description("Epoch: "+str(epoch)+" Avg loss of entire dataset: "+str(train_loss))

		for batch_idx,(data,target) in progress_bar:
			data = data.to(device)
			target = target.to(device)

			h_target = target.view(-1,1)
			h_target = utils.to_categorical(h_target,num_classes)
			h_data = data.view(-1,np.prod(data.size()[1:]))

			output,hidden_acts = model(data)

			for i in range(len(hidden_acts)):
				output, hidden_acts = model(data)
				exec('optimizer_'+str(i)+".zero_grad()")

				hidden_acts[i] = hidden_acts[i].view(-1,np.prod(hidden_acts[i].size()[1:]))
				hsic_zx, hsix_yz = math_funcs.HSIC_objective(hidden_acts[i],h_data,h_target,sigma)

				loss = hsic_zx - beta*hsix_yz
				loss.backward()

				exec('optimizer_'+str(i)+'.step()')
				exec('optimizer_'+str(i)+'.zero_grad()')

			del data
			del target
			del hidden_acts
			del h_target
			del h_data
			del hsix_yz
			del hsic_zx
			del output
			del loss
			torch.cuda.empty_cache()

		train_accuracy, train_loss = utils.compute_model_accuracy(train_dataloader,dnn_model,model_type)
		validation_accuracy,validation_loss = utils.compute_model_accuracy(val_dataloader,dnn_model,model_type)
		tensorboard_writer.add_scalar("Validation loss",(validation_loss),global_step=val_step)
		val_step += 1

		print("\nEpoch:               "+str(epoch))
		print("Training Loss:       "+str(train_loss))
		print("Training Accuracy:   "+str(train_accuracy))
		print("Validation Loss:     "+str(validation_loss))
		print("Validation Accuracy: "+str(validation_accuracy)+"\n")

		# Save the weights at the end of the each epoch
		checkpoint_file = "HSIC_" + model_type+"_"+dataset_type+"_"+"_Checkpoint_"+str(epoch)+".pt"
		
		checkpoint = {}
		checkpoint["model"] = dnn_model.state_dict()
		checkpoint["train_loss"] = train_loss
		checkpoint["validation_loss"] = validation_loss
		checkpoint["test_loss"] = test_loss
		checkpoint["tensorboard_train_step"] = train_step
		checkpoint["tensorboard_val_step"] = val_step
		save_dir = os.path.join(checkpoint_dir,checkpoint_file)
		torch.save(checkpoint,save_dir)

		del checkpoint
		torch.cuda.empty_cache()

def sample_deletion_training(dataset_type,dataset_dir,checkpoint_dir,model_type,batch_size,device,epochs,last_checkpoint,tensorboard_writer,lr,momentum):
	''' The sample delete method of meta cognitive training '''

	# Initialize and load the dataloader
	if dataset_type == "MNIST":
		train_dataloader,val_dataloader,test_datalaoder = dataset.load_mnist(dataset_dir,batch_size)
	elif dataset_type == "CIFAR10":
		train_dataloader,val_dataloader,test_datalaoder = dataset.load_cifar10(dataset_dir,batch_size)
	else:
		print("Dataset type not supported")
		return

	# Initialize the model
	if model_type == "qnn_Linear":
		dnn_model = qnn_model.LinearModel(dataset_type).to(device)
	elif model_type == "qnn_LeNet":
		dnn_model = qnn_model.LeNet(dataset_type).to(device)
	elif model_type == "Linear":
		dnn_model = model.LinearModel(dataset_type).to(device)
	elif model_type == "LeNet":
		dnn_model = model.LeNet(dataset_type).to(device)
	else:
		print("Model type not supported")
		return

	# Initialize the optimizer
	optimizer = torch.optim.SGD(dnn_model.parameters(),lr=lr,momentum=momentum)
	loss_criterion =  torch.nn.BCELoss().to(device)

	cur_epoch = 0
	train_step = 0
	val_step = 0

	train_loss = 0.
	test_loss = 0.
	validation_loss = 0.
	validation_accuracy = 0.

	# Meta-Cognitive parameters
	initial_threshold = 1.
	rate_of_threshold_decay = -0.0347

	# Load a previous checkpoint to resume training
	if last_checkpoint:
		dnn_model.load_state_dict(last_checkpoint["model"])
		optimizer.load_state_dict(last_checkpoint["optimizer"])
		cur_epoch = last_checkpoint["epoch"]
		train_loss = last_checkpoint["train_loss"]
		validation_loss = last_checkpoint["validation_loss"]
		test_loss = last_checkpoint["test_loss"]
		train_step = last_checkpoint["tensorboard_train_step"]
		val_step = last_checkpoint["tensorboard_val_step"]

		#Initialize the new lr to optimizer
		for groups in optimizer.param_groups:
			groups["lr"] = lr

	# Start Training
	for epoch in range(cur_epoch,epochs):

		progress_bar = tqdm.tqdm(enumerate(train_dataloader))
		dnn_model.train()
		loss_criterion.train()
		progress_bar.set_description("Epoch: "+str(epoch)+" Avg loss of entire dataset: "+str(train_loss))

		current_threshold = initial_threshold*torch.exp(-rate_of_threshold_decay*epoch)

		for batch_idx,(data,target) in progress_bar:
			data = data.to(device)
			target = utils.to_categorical(target)
			target = target.to(device)

			output, _ = dnn_model(data)

			output = utils.selective_sampling_outputs(output,target,current_threshold)

			loss = loss_criterion(output,target)
			loss.backward()
			progress_bar.set_description("Epoch: "+str(epoch)+" Batch index: "+str(batch_idx)+" Batch-wise Training Loss: "+str(loss.item()))

			optimizer.step()
			optimizer.zero_grad()

			train_loss = (train_loss*(batch_idx) + loss.item())/(batch_idx+1)
			tensorboard_writer.add_scalar("Batch-wise Training Loss",(loss.item()),global_step=train_step)
			train_step += 1
			del data
			del target
			del output
			del loss
			torch.cuda.empty_cache()

		train_accuracy, train_loss = utils.compute_model_accuracy(train_dataloader,dnn_model,model_type)
		validation_accuracy,validation_loss = utils.compute_model_accuracy(val_dataloader,dnn_model,model_type)
		tensorboard_writer.add_scalar("Validation loss",(validation_loss),global_step=val_step)
		val_step += 1

		print("\nEpoch:               "+str(epoch))
		print("Training Loss:       "+str(train_loss))
		print("Training Accuracy:   "+str(train_accuracy))
		print("Validation Loss:     "+str(validation_loss))
		print("Validation Accuracy: "+str(validation_accuracy)+"\n")

		# Save the weights at the end of the each epoch
		checkpoint_file = "Sample_deletion_" + model_type+"_"+dataset_type+"_"+"_Checkpoint_"+str(epoch)+".pt"
		
		checkpoint = {}
		checkpoint["model"] = dnn_model.state_dict()
		checkpoint["optimizer"] = optimizer.state_dict()
		checkpoint["train_loss"] = train_loss
		checkpoint["validation_loss"] = validation_loss
		checkpoint["test_loss"] = test_loss
		checkpoint["tensorboard_train_step"] = train_step
		checkpoint["tensorboard_val_step"] = val_step
		save_dir = os.path.join(checkpoint_dir,checkpoint_file)
		torch.save(checkpoint,save_dir)

		del checkpoint
		torch.cuda.empty_cache()


def main():
	model_type = "Linear"
	dataset_type = "MNIST"
	dataset_dir = "./datasets"
	checkpoint_dir = "./weights"
	batch_size = 128
	device = "cuda"
	epochs = 20
	last_checkpoint = None
	lr = 0.01
	momentum = 0.9
	tensorboard_writer = SummaryWriter(f'logs/loss')

	train_type = "standard"

	if train_type == "standard":
		train(dataset_type,dataset_dir,checkpoint_dir,model_type,batch_size,device,epochs,last_checkpoint,tensorboard_writer,lr,momentum)
	elif train_type == "hsic":
		hsic_train(dataset_type,dataset_dir,checkpoint_dir,model_type,batch_size,device,epochs,last_checkpoint,tensorboard_writer,lr,momentum)
	elif train_type == "sample_deletion":
		sample_deletion_training(dataset_type,dataset_dir,checkpoint_dir,model_type,batch_size,device,epochs,last_checkpoint,tensorboard_writer,lr,momentum)

if __name__ == '__main__':
	main()
