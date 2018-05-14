import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

	def __init__(self, feature_layers, classifier):
		super(Model, self).__init__()
		self.feature_layers = feature_layers
		self.classifier = classifier

	def forward(self, input):
		input = self.feature_layers(input)
		input = input.view(input.size(0), -1)
		input = self.classifier(input)
		return input

def make_layers(layout):
	layers = []
	for layer in layout:
		if layer[0] == 'A':
			layers += [nn.AvgPool2d(kernel_size= (layer[1][0], layer[1][1]), stride= layer[2], padding= layer[3])]
		elif layer[0] == 'M':
			layers += [nn.MaxPool2d(kernel_size= (layer[1][0], layer[1][1]), stride= layer[2], padding= layer[3])]
		elif layer[0] == 'C':
			conv2d = nn.Conv2d(in_channels= layer[1], out_channels= layer[2], 
				kernel_size= (layer[3][0], layer[3][1]), stride= layer[4], padding= layer[5])
			if layer[6] is True and layer[7] is True:
				layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.SELU(inplace=True)]
			elif layer[6] is True and layer[7] is not True:
				layers += [conv2d, nn.SELU(inplace= True)]
			else:
				layers += [conv2d]

	return nn.Sequential(*layers)

def make_classifier_layers(layout):
	layers = []
	for layer in layout:
		if layer[0] == 'L':
			layers += [nn.Linear(layer[1], layer[2]), nn.BatchNorm2d(layer[2]), nn.SELU(inplace= True)]
		elif layer[0] == 'D':
			layers += [nn.Dropout(layer[1])]
		elif layer[0] == 'FC':
			layers += [nn.Linear(layer[1], layer[2])]

	return nn.Sequential(*layers)

