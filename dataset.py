import torch
from torchvision import datasets,transforms

def load_mnist(folder_path,batch_size):
	'''Load the MNIST dataset'''
	train_data = datasets.MNIST(folder_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
	test_data = datasets.MNIST(folder_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]))

	kwargs = {'num_workers': 4, 'pin_memory': True}

	train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
	test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)

	return train_data_loader, test_data_loader

def load_cifar10(dataset_dir,batch_size):
	'''
		Load the train and test data laoder for cifar 10 dataset
	'''
	train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.49139968, 0.48215841, 0.44653091),(0.24703223, 0.24348513, 0.26158784))])
	test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.49421428, 0.48513139, 0.45040909),(0.24665252, 0.24289226, 0.26159238))])

	train_dataset = torchvision.datasets.CIFAR10(dataset_dir,train=True,download=True,transform=train_transform)
	test_dataset = torchvision.datasets.CIFAR10(dataset_dir,train=False,download=True,transform=test_transform)

	kwargs = {"num_workers":4,"pin_memory":True}
	train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,**kwargs)
	test_datalaoder = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,**kwargs)

	return train_dataloader,test_datalaoder