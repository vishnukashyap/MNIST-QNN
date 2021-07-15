import os
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

import model
import dataset

def train(dataset_type,dataset_dir,checkpoint_dir,model_type,batch_size,device,epochs,last_checkpoint,tensorboard_writer,lr,momentum):
	''' Standard training with back propagation'''

	# Initialize and load the dataloader
	if dataset_type == "MNIST":
		train_dataloader,test_datalaoder = dataset.load_mnist(dataset_dir,batch_size)
	elif dataset_type == "CIFAR10":
		train_dataloader,test_datalaoder = dataset.load_cifar10(dataset_dir,batch_size)
	else:
		print("Dataset type not supported")
		return

	# Initialize the model
	if model_type == "Linear":
		qnn_model = model.LinearModel().to(device)
	elif model_type == "LeNet":
		qnn_model = model.LeNet().to(device)
	else:
		print("Model type not supported")
		return

	# Initialize the optimizer and loss criterion
	optimizer = torch.optim.SGD(qnn_model.parameters(),lr=lr,momentum=momentum)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[7,14],gamma=0.1,verbose=True)
	loss_criterion =  torch.nn.CrossEntropyLoss().to(device)

	cur_epoch = 0
	train_step = 0
	val_step = 0

	train_loss = 0.
	test_loss = 0.
	validation_loss = 0.

	# Load a previous checkpoint to resume training
	if last_checkpoint != None:
		qnn_model.load_state_dict(last_checkpoint["model"])
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
		qnn_model.train()
		loss_criterion.train()
		progress_bar.set_description("Epoch: "+str(epoch)+" Avg loss of entire dataset: "+str(train_loss))
		for batch_idx,(data,target) in progress_bar:
			data = data.to(device)
			target = target.to(device)

			output = qnn_model(data)
			loss = loss_criterion(output,target)
			loss.backward()
			progress_bar.set_description("Epoch: "+str(epoch)+" Batch index: "+str(batch_idx)+" Batch-wise Training Loss: "+str(loss.item()))

			optimizer.step()
			optimizer.zero_grad()
			# qnn_model.clip_weights(-1,1)

			train_loss = (train_loss*(batch_idx) + loss.item())/(batch_idx+1)
			tensorboard_writer.add_scalar("Batch-wise Training Loss",(loss.item()),global_step=train_step)
			train_step += 1
			del data
			del target
			del output
			del loss
			torch.cuda.empty_cache()
		# validation_loss = compute_validation_loss()
		# tensorboard_writer.add_scalar("Validation loss",(validation_loss),global_step=val_step)
		# val_step += 1

		# Save the weights at the end of the each epoch
		checkpoint_file = model_type+"_"+dataset_type+"_"+"_Checkpoint_"+str(epoch)+".pt"
		
		checkpoint = {}
		checkpoint["model"] = qnn_model.state_dict()
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

		scheduler.step()

def main():
	model_type = "Linear"
	dataset_type = "MNIST"
	dataset_dir = "./datasets"
	checkpoint_dir = "./weights"
	batch_size = 128
	device = "cuda"
	epochs = 20
	last_checkpoint = None
	lr = 0.1
	momentum = 0.9
	tensorboard_writer = SummaryWriter(f'logs/loss')
	train(dataset_type,dataset_dir,checkpoint_dir,model_type,batch_size,device,epochs,last_checkpoint,tensorboard_writer,lr,momentum)

if __name__ == '__main__':
	main()
