import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(Discriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		
		## last layer
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		
		## up-sampling : produce a pixel-wise classification of an image.
		self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		x = self.up_sample(x)
		#x = self.sigmoid(x) 

		return x
	
	## I don't understand what for
	def train_params(self, requires_grad=True):
    	# Function to set requires_grad attribute of parameters for training
    		for param in self.parameters():
        		param.requires_grad = requires_grad
