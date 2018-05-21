# Model Architecture:
# Convolution: ['C', in_channels, out_channels, (kernel), stride, padding, Activation Function]
# Max Pooling: ['M', (kernel), stride, padding]
# Average Pooling: ['A', (kernel), stride, padding]
# Linear Layer: ['L', in_features, out_features, Activation Function]
# Dropout : ['D', probability]
# Alpha Dropout : ['AD', probability]
# Classifying layer: ['FC', in_features, num_classes]
# Possible Activation Fns: 'ReLU', 'PReLU', 'SELU', 'LeakyReLU', 'None'->(Contains no Batch Norm for dimensionality reduction 1x1 kernels)

feature_layers = {
	
	'1': [[[], [], []], 
		[[], [], []], 
		[[], [], []], 
		[[], [], []]]
}