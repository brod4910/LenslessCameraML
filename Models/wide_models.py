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
	
	'1':[
		[[['C', 1, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 'ReLU'], ['M', (2,2), 2, 0]]],
		[[['C', 512, 128, (1,1), 1, 0, 'None'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 392, (3,3), 1, 1, 'ReLU']], [['C', 512, 128, (1,1), 1, 0, 'None'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 392, (3,3), 1, 1, 'ReLU']], [['C', 512, 128, (1,1), 1, 0, 'None'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 392, (3,3), 1, 1, 'ReLU']]], 
		[[['C', 1176, 256, (1,1), 1, 0, 'None'], ['C', 256, 512, (3,3), 1, 1, 'ReLU']], [], []], 
		[[], [], []], 
		[[], [], []]
		]
}