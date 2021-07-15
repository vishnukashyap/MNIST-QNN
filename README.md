# MNIST-QNN
Training a Simple Quantized Neural Network using Pytorch and brevitas

Packages required:
	- pytorch
	- torchvision
	- brevitas
	- tqdm
	- tensorboard

Folder Structure:
	root
		- dataset
			- MNIST (will be downloaded by torch if it's not already preset)
			- CIFAR10 (will be downloaded by torch if it's not already preset)
		- logs
			- loss (contains the tensorboard logs to be visualized later)
		- weights (contains all the trained weights stored at the end of every epoch)
