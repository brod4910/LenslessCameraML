import torch
import torch.nn as nn
import torch.nn.functional as F

class Wide_Model(nn.Module):

    def __init__(self, feature_layers, classifier, device):
        super(Wide_Model, self).__init__()
        self.feature_layers = feature_layers
        self.classifier = classifier
        self.device = device

    def forward(self, input):
        for section in self.feature_layers:
            outputs = []
            for branch in section:
                branch = branch.to(self.device)
                output = branch(input)
                outputs += output
            if len(outputs) == 1:
                input = outputs[0]
            else:
                input = torch.cat(outputs, 1)
        # input = self.feature_layers(input)
        input = input.view(input.size(0), -1)
        input = self.classifier(input)
        return input

def make_wide_layers(layout):
    sections = []
    for section in layout:
        branches = []
        for branch in section:
            layers = []
            for layer in branch:
                if layer[0] == 'A':
                    layers += [nn.AvgPool2d(kernel_size= (layer[1][0], layer[1][1]), stride= layer[2], padding= layer[3])]
                elif layer[0] == 'M':
                    layers += [nn.MaxPool2d(kernel_size= (layer[1][0], layer[1][1]), stride= layer[2], padding= layer[3])]
                elif layer[0] == 'C':
                    conv2d = nn.Conv2d(in_channels= layer[1], out_channels= layer[2], 
                        kernel_size= (layer[3][0], layer[3][1]), stride= layer[4], padding= layer[5])
                    if layer[6] == 'ReLU':
                        layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.ReLU(inplace=True)]
                    elif layer[6] == 'PReLU':
                        layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.PReLU(inplace=True)]
                    elif layer[6] == 'SELU':
                        layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.SELU(inplace=True)]
                    elif layer[6] == 'LeakyReLU':
                        layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.LeakyReLU(inplace=True)]
                    else:
                        layers += [conv2d]
            branches += [nn.Sequential(*layers)]
        sections += [branches]
    return sections

def make_classifier_layers(layout):
    layers = []
    for layer in layout:
        if layer[0] == 'L':
            if layer[3] == 'ReLU':
                layers += [nn.Linear(layer[1], layer[2]), nn.BatchNorm1d(layer[2]), nn.ReLU(inplace= True)]
            elif layer[3] == 'PReLU':
                layers += [nn.Linear(layer[1], layer[2]), nn.BatchNorm1d(layer[2]), nn.PReLU(inplace= True)]
            elif layer[3] == 'SELU':
                layers += [nn.Linear(layer[1], layer[2]), nn.BatchNorm1d(layer[2]), nn.SELU(inplace= True)]
            elif layer[3] == 'LeakyReLU':
                layers += [nn.Linear(layer[1], layer[2]), nn.BatchNorm1d(layer[2]), nn.LeakyReLU(inplace= True)]
        elif layer[0] == 'D':
            layers += [nn.Dropout(layer[1])]
        elif layer[0] == 'AD':
            layers+= [nn.AlphaDropout(layer[1])]
        elif layer[0] == 'FC':
            layers += [nn.Linear(layer[1], layer[2])]

    return nn.Sequential(*layers)
