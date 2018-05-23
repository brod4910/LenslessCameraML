# Model Architecture:
# Convolution: ['C', in_channels, out_channels, (kernel), stride, padding, Activation Function]
# Max Pooling: ['M', (kernel), stride, padding]
# Average Pooling: ['A', (kernel), stride, padding]
# Linear Layer: ['L', in_features, out_features, Activation Function]
# Dropout : ['D', probability]
# Alpha Dropout : ['AD', probability]
# Classifying layer: ['FC', in_features, num_classes]
# Possible Activation Fns: 'ReLU', 'PReLU', 'SELU', 'LeakyReLU', 'None'->(Contains no Batch Norm for dimensionality reduction 1x1 kernels)

# 'branch_1': ,
# 'branch_2': ,
# 'branch_3':

feature_layers = {
	
	'1': 
		{
		'section_1': 
					{
						'branch_1': [['C', 1, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 'ReLU'], ['M', (2,2), 2, 0]]
					}
		'section_2': 
					{
						'branch_1': [['C', 512, 128, (1,1), 1, 0, 'None'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 392, (3,3), 1, 1, 'ReLU']],
						'branch_2': [['C', 512, 128, (1,1), 1, 0, 'None'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 392, (3,3), 1, 1, 'ReLU']],
						'branch_3': [['C', 512, 128, (1,1), 1, 0, 'None'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 392, (3,3), 1, 1, 'ReLU']]
					}
		'section_3': 
					{
						'branch_1': [['C', 1176, 256, (1,1), 1, 0, 'None'], ['C', 256, 512, (3,3), 1, 1, 'ReLU']],
						'branch_2': [['C', 1176, 392, (1,1), 1, 0, 'None'], ['C', 392, 496, (5,5), 1, 2, 'ReLU']],
						'branch_3': [['C', 1176, 256, (1,1), 1, 0, 'None'], ['C', 256, 384, (3,3), 1, 1, 'ReLU']]
					}
		'section_4': 
					{
						'branch_1': [['M', (3,3), 2, 0], ['C', 1392, 384, (1,1), 1, 0, 'None']],
						'branch_2': [['C', 1392, 384, (1,1), 1, 0, 'None'], ['C', 384, 384, (3,3), 2, 0, 'ReLU']],
						'branch_3': [['C', 1392, 384, (1,1), 1, 0, 'None'], ['C', 384, 384, (3,3), 2, 0, 'ReLU']]
					},
		'section_5': 
					{
						'branch_1': [['C', 1152, 256, (1,1), 1, 0, 'None'], ['C', 256, 512, (3,3), 1, 1, 'ReLU']],
						'branch_2': [['C', 1152, 392, (1,1), 1, 0, 'None'], ['C', 392, 496, (5,5), 1, 2, 'ReLU']],
						'branch_3': [['C', 1152, 256, (1,1), 1, 0, 'None'], ['C', 256, 384, (3,3), 1, 1, 'ReLU']]
					},
		'section_6': 
					{
						'branch_1': [['M', (2,2), 2, 0], ['C', 1392, 392, (1,1), 1, 0, 'None']]
					}
		}
}

classifier_layers = {
	
	'1': [['L', 392 * 13 * 13, 1024, 'ReLU'], ['D', .7], ['L', 1024, 2048, 'ReLU'], ['D', .7], ['FC', 2048, 10]]
}