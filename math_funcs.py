import torch
import numpy as np

def sigma_estimation(X, Y):
	""" sigma from median distance
	"""
	D = disMat(torch.cat([X,Y]))
	D = D.detach().cpu().numpy()
	Itri = np.tril_indices(D.shape[0], -1)
	Tri = D[Itri]
	med = np.median(Tri)
	if med <= 0:
		med=np.mean(Tri)
	if med<1E-2:
		med=1E-2
	return med

def disMat(X):
	'''
		Eucledian distance matrix
	'''
	r = torch.sum(X*X,1) # This matrix would be batch_size x all values in one dimension
	r = r.view(-1,1)
	a = torch.mm(X,torch.transpose(X,0,1)) # This matrix would have dimensions batch_size x batch_size
	# distance matrix = X^2 -2xXxY - Y^2 => (X-Y)^2
	D = r.expand_as(a) -2*a + torch.transpose(r,0,1).expand_as(a)
	D = torch.abs(D)
	return D

def kernelMat(X,sigma):
	'''
		This function calculates and returns the value obtained by applying a kernel which is sampled from an RKHS
		on continuous random variable which are input, target and intermediate activation
	'''
	m = int(X.size()[0])
	dim = int(X.size()[1])
	H = torch.eye(m) - (1./m)*torch.ones([m,m])
	D = disMat(X)

	if sigma:
		variance = 2.*sigma*sigma*dim
		Kx = torch.exp(-D/variance).type(torch.FloatTensor)
		Kxc = torch.mm(Kx,H) # This is the centerned matrix which is used for the calculation of nHSIC
		return Kxc
	else:
		try:
			sx = sigma_estimation(X,X)
			Kx = torch.exp( -D / (2.*sx*sx)).type(torch.FloatTensor)
		except RuntimeError as e:
			raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
				sx, torch.max(X), torch.min(X)))
		Kxc = torch.mm(Kx,H)
		return Kxc

def nHSIC(X,Y,sigma):
	'''
		normalised HSIC calculation as per the formula given in the paper
	'''
	m = int(X.size()[0])
	Kxc = kernelMat(X,sigma)
	Kyc = kernelMat(Y,sigma)

	epsilon = 1E-5
	I = torch.eye(m)
	Kxc_i = torch.inverse(Kxc + epsilon*m*I)
	Kyc_i = torch.inverse(Kyc + epsilon*m*I)
	Rx = (Kxc.mm(Kxc_i))
	Ry = (Kyc.mm(Kyc_i))
	Pxy = torch.sum(torch.mul(Rx,Ry.t()))

	return Pxy

def HSIC_objective(hidden,data,target,sigma):
	'''
		The objective value to calculate the gradient value to update the wieghts
	'''
	hsic_zx = nHSIC(hidden,data,sigma)
	hsix_yz = nHSIC(hidden,target,sigma)

	return hsic_zx,hsix_yz