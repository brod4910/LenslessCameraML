import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.checkpoint import checkpoint

'''
    Creates a model with feature layers, and classifying layers. The model uses
    a dummy model to tie the input to dummy tensor so if checkpointing is used, the model
    won't have zero gradients at its input. This is a must if the model is checkpointed.
'''
class Model(nn.Module):

    def __init__(self, feature_layers, classifier, checkpoint= False):
        super(Model, self).__init__()
        self.checkpoint = checkpoint
        self.feature_layers = feature_layers
        self.classifier = classifier
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.module_wrapper = ModuleWrapperIgnores2ndArg(self.feature_layers)

    def forward(self, input_var):
        if self.checkpoint is True:
            # x = checkpoint(self.module_wrapper,x,self.dummy_tensor)
            input_var = checkpoint(self.module_wrapper, input_var, self.dummy_tensor)
        else:
            input_var = self.feature_layers(input_var)

        input_var = input_var.view(input_var.size(0), -1)
        input_var = self.classifier(input_var)
        return input_var

'''
    Model wrapper that is used when checkpointing. The model takes the desired feature layers
    and performs the forward pass for the feature layers. This is so that the model doesn't have
    zero gradients at its input.
'''
class ModuleWrapperIgnores2ndArg(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self,x, dummy_arg=None):
        assert dummy_arg is not None
        x = self.module(x)
        return x

'''
    Creates the model layers depending on the arrays of layers that are passed into layout.
    An array layer is as follows:
        Convolution: ['C', in_channels, out_channels, (kernel), stride, dilation, padding, Activation Function]
        Max Pooling: ['M', (kernel), stride, padding]
        Average Pooling: ['A', (kernel), stride, padding]
        Possible Activation Fns: 'ReLU', 'ReLU_NoB2d', 'Sigmoid', 'Sigmoid_NoB2d','PReLU', 'SELU', 'LeakyReLU', 'None'->(Contains no Batch Norm for dimensionality reduction 1x1 kernels)
'''
def make_layers(layout, upsample= False):
    layers = []

    if len(layout) == 0:
        return None

    for layer in layout:
        if layer[0] == 'A':
            layers += [nn.AvgPool2d(kernel_size= (layer[1][0], layer[1][1]), stride= layer[2], padding= layer[3])]
        elif layer[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size= (layer[1][0], layer[1][1]), stride= layer[2], padding= layer[3], return_indices= upsample)]
        elif layer[0] == 'C':
            conv2d = nn.Conv2d(in_channels= layer[1], out_channels= layer[2], 
                kernel_size= (layer[3][0], layer[3][1]), stride= layer[4], dilation= layer[5], padding= layer[6])
            if layer[7] == 'ReLU_NoB2d':
                layers += [conv2d, nn.ReLU(inplace= True)]
            elif layer[7] == 'ReLU':
                layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.ReLU(inplace= True)]
            elif layer[7] == 'PReLU':
                layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.PReLU(inplace= True)]
            elif layer[7] == 'SELU':
                layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.SELU(inplace= True)]
            elif layer[7] == 'LeakyReLU':
                layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.LeakyReLU(inplace= True)]
            elif layer[7] == 'Sigmoid':
                layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.Sigmoid()]
            elif layer[7] == 'Sigmoid_NoB2d':
                layers += [conv2d, nn.Sigmoid()]
            elif layer[7] == 'Tanh':
                layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.Tanh()]
            elif layer[7] == 'Tanh_NoB2d':
                layers += [conv2d, nn.Tanh()]
            else:
                layers += [conv2d]
        elif layer[0] == 'CTrans':
            conv2d = nn.ConvTranspose2d(in_channels= layer[1], out_channels= layer[2], 
                kernel_size= (layer[3][0], layer[3][1]), stride= layer[4], dilation= layer[5], padding= layer[6])
            if layer[7] == 'ReLU_NoB2d':
                layers += [conv2d, nn.ReLU(inplace= True)]
            elif layer[7] == 'ReLU':
                layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.ReLU(inplace= True)]
        elif layer[0] == 'MUnP':
            layers += [nn.MaxUnpool2d(kernel_size= (layer[1][0], layer[1][1]), stride= layer[2], padding= layer[3])]
        elif layer[0] == 'D2d':
            layers += [nn.Dropout2d(layer[1])]
        elif layer[0] == 'D':
            layers += [nn.Dropout(layer[1])]

    return nn.Sequential(*layers)

'''
    Creates the classifying layers depending on the arrays of layers that are passed into layout.
    An array layer is as follows:
        Linear Layer: ['L', in_features, out_features, Activation Function]
        Dropout : ['D', probability]
        Dropout 2D : ['D2d', propability]
        Alpha Dropout : ['AD', probability]
        Classifying layer: ['FC', in_features, num_classes]
        Possible Activation Fns: 'ReLU', 'PReLU', 'SELU', 'LeakyReLU', 'None'->(Contains no Batch Norm for dimensionality reduction 1x1 kernels)
'''
def make_classifier_layers(layout):
    layers = []
    for layer in layout:
        if layer[0] == 'L':
            if layer[3] == 'ReLU_NoB2d':
                layers += [nn.Linear(layer[1], layer[2]), nn.ReLU(inplace= True)]
            elif layer[3] == 'ReLU':
                layers += [nn.Linear(layer[1], layer[2]), nn.BatchNorm1d(layer[2]), nn.ReLU(inplace= True)]
            elif layer[3] == 'PReLU':
                layers += [nn.Linear(layer[1], layer[2]), nn.BatchNorm1d(layer[2]), nn.PReLU(inplace= True)]
            elif layer[3] == 'SELU':
                layers += [nn.Linear(layer[1], layer[2]), nn.BatchNorm1d(layer[2]), nn.SELU(inplace= True)]
            elif layer[3] == 'LeakyReLU':
                layers += [nn.Linear(layer[1], layer[2]), nn.BatchNorm1d(layer[2]), nn.LeakyReLU(inplace= True)]
            elif layer[3] == 'Tanh_NoB2d':
                layers += [nn.Linear(layer[1], layer[2]), nn.Tanh()]
            elif layer[3] == 'Tanh':
                layers += [nn.Linear(layer[1], layer[2]), nn.BatchNorm1d(layer[2]) ,nn.Tanh()]
        elif layer[0] == 'D':
            layers += [nn.Dropout(layer[1])]
        elif layer[0] == 'AD':
            layers+= [nn.AlphaDropout(layer[1])]
        elif layer[0] == 'FC':
            layers += [nn.Linear(layer[1], layer[2])]

    return nn.Sequential(*layers)

