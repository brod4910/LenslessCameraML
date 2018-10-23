# Model Architecture:
# Convolution: ['C', in_channels, out_channels, (kernel), stride, dilation, padding, Activation Function]
# Max Pooling: ['M', (kernel), stride, padding]
# Average Pooling: ['A', (kernel), stride, padding]
# Linear Layer: ['L', in_features, out_features, Activation Function]
# Dropout : ['D', probability]
# Dropout 2D : ['D2d', propability]
# Alpha Dropout : ['AD', probability]
# Classifying layer: ['FC', in_features, num_classes]
# Possible Activation Fns: 'ReLU', 'PReLU', 'SELU', 'LeakyReLU', 'None'->(Contains no Batch Norm for dimensionality reduction 1x1 kernels)
# srun python main.py --batch-size 16 --epochs 50 --lr 0.001 --momentum .9 --log-interval 100 --root-dir ../ --train-input-file ../clipped_training_data --train-target-file ../clipped_training_targets --test-input-file ../clipped_test_data --test-target-file ../clipped_test_targets

# The calculations below are constrained to stride of 1
# padding of 2 for 3x3 dilated convolution of 2 for same input/output image size
# padding of 3 for 3x3 dilated convolution of 3
#
# padding of 4 for 5x5 dilated convolution of 2 for same input/output image size
# padding of 6 for 5x5 dilated convolution of 3
#
# padding of 6 for 7x7 dilated convolution of 2 for same input/output image size
# padding of 9 for 7x7 dilated convolution of 3


feature_layers = {

    '1': [['C', 1, 128, (3,3), 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0], ['C', 256, 512, (3,3), 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0], ['C', 512, 256, (3,3), 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0]],

    # 92% accuracy on digits 0-9 with .001 lr and .9 momentum
    # 82% accuracy with .01 lr and .9 momentum
    '2': [['C', 1, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 'ReLU'], ['C', 512, 128, (1,1), 1, 0, 'None'], 
    ['M', (2,2), 2, 0], ['C', 128, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 'ReLU'], ['C', 512, 128, (1,1), 1, 0, 'None'], 
    ['M', (2,2), 2, 0], ['C', 128, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 'ReLU'], ['C', 512, 128, (1,1), 1, 0, 'None'],
    ['M', (2,2), 2, 0], ['C', 128, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0]],

    # same as model two but reduces dimensionality after maxpooling
    # ['C', 1, 128, (3,3), 1, 1, 'ReLU']
    '2.5': [['C', 1, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0], ['C', 512, 128, (1,1), 1, 0, 'None'], ['C', 128, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0], ['C', 512, 128, (1,1), 1, 0, 'None'], ['C', 128, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 'ReLU'],
    ['M', (2,2), 2, 0], ['C', 512, 128, (1,1), 1, 0, 'None'], ['C', 128, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0]],

    '2.5.5': [['C', 1, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0], ['C', 512, 128, (1,1), 1, 0, 'ReLU'], ['C', 128, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0], ['C', 512, 128, (1,1), 1, 0, 'ReLU'], ['C', 128, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 'ReLU'],
    ['M', (2,2), 2, 0], ['C', 512, 128, (1,1), 1, 0, 'ReLU'], ['C', 128, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0]],

    # 91% accuracy on digits 0-9
    '3': [['C', 1, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 512, (5,5), 1, 2, 'ReLU'], 
    ['M', (2,2), 2, 0], ['C', 512, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 512, (5,5), 1, 2, 'ReLU'],
    ['M', (2,2), 2, 0], ['C', 512, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 512, (5,5), 1, 2, 'ReLU'],
    ['M', (2,2), 2, 0], ['C', 512, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 512, (5,5), 1, 2, 'ReLU'],
    ['M', (2,2), 2, 0]],

    # 91% accuracy on digits 0-9
    '4': [['C', 1, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (5,5), 1, 2, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0], ['C', 512, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (5,5), 1, 2, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 'ReLU'],
    ['M', (2,2), 2, 0], ['C', 512, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (5,5), 1, 2, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 'ReLU'],
    ['M', (2,2), 2, 0], ['C', 512, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (5,5), 1, 2, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 'ReLU'],
    ['M', (2,2), 2, 0]],

    # 90% accuracy on digits 0-9
    '5': [['C', 1, 64, (3,3), 1, 1, 'ReLU'], ['C', 64, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 128, (1,1), 1, 0, 'ReLU'], ['C', 128, 256, (3,3), 2, 0, 'ReLU'],
    ['C', 256, 64, (3,3), 1, 1, 'ReLU'], ['C', 64, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 128, (1,1), 1, 0, 'ReLU'], ['M', (2,2), 2, 0],
    ['C', 128, 64, (3,3), 1, 1, 'ReLU'], ['C', 64, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 128, (1,1), 1, 0, 'ReLU'], ['C', 128, 256, (3,3), 2, 0, 'ReLU'],
    ['C', 256, 64, (3,3), 1, 1, 'ReLU'], ['C', 64, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], ['C', 256, 128, (1,1), 1, 0, 'ReLU'], 
    ['M', (2,2), 2, 0]],

    # 85% accuracy on digits 0-9
    '6': [['C', 1, 32, (3,3), 1, 1, 'SELU'], ['C', 32, 128, (3,3), 1, 1, 'SELU'], ['C', 128, 256, (3,3), 1, 1, 'SELU'], ['C', 256, 128, (1,1), 1, 0, 'None'], ['C', 128, 256, (3,3), 2, 0, 'SELU'],
    ['C', 256, 256, (3,3), 1, 1, 'SELU'], ['C', 256, 128, (3,3), 1, 1, 'SELU'], ['C', 128, 256, (3,3), 1, 1, 'SELU'], ['M', (2,2), 2, 0], ['C', 256, 128, (1,1), 1, 0, 'None'],
    ['C', 128, 128, (3,3), 1, 1, 'SELU'], ['C', 128, 256, (3,3), 1, 1, 'SELU'], ['C', 256, 256, (3,3), 1, 1, 'SELU'], ['C', 256, 128, (1,1), 1, 0, 'None'], ['C', 128, 256, (3,3), 2, 0, 'SELU'],
    ['C', 256, 128, (3,3), 1, 1, 'SELU'], ['C', 128, 128, (3,3), 1, 1, 'SELU'], ['C', 128, 256, (3,3), 1, 1, 'SELU'], ['M', (2,2), 2, 0]],

    # 89% accuracy on digits 0-9 (without 4 max pooling layers) ... 88% accuracy on digits 0-9 (with 4 max pooling layers)
    '7': [['C', 1, 128, (3,3), 1, 1, 'SELU'], ['C', 128, 128, (3,3), 1, 1, 'SELU'], ['C', 128, 256, (3,3), 1, 1, 'SELU'], ['C', 256, 256, (3,3), 1, 1, 'SELU'], ['C', 256, 256, (3,3), 1, 1, 'SELU'], ['C', 256, 64, (1,1), 1, 0, 'None'], ['C', 64, 128, (3,3), 1, 1, 'SELU'], ['M', (2,2), 2, 0],
    ['C', 128, 128, (3,3), 1, 1, 'SELU'], ['C', 128, 256, (3,3), 1, 1, 'SELU'], ['C', 256, 256, (3,3), 1, 1, 'SELU'], ['C', 256, 256, (3,3), 1, 1, 'SELU'], ['M', (2,2), 2, 0],
    ['C', 256, 128, (3,3), 1, 1, 'SELU'], ['C', 128, 256, (3,3), 1, 1, 'SELU'], ['C', 256, 256, (3,3), 1, 1, 'SELU'], ['C', 256, 256, (3,3), 1, 1, 'SELU'], ['C', 256, 64, (1,1), 1, 0, 'None'], ['C', 64, 128, (3,3), 1, 1, 'SELU'], ['M', (2,2), 2, 0],
    ['C', 128, 128, (3,3), 1, 1, 'SELU'], ['C', 128, 256, (3,3), 1, 1, 'SELU'], ['C', 256, 256, (3,3), 1, 1, 'SELU'], ['C', 256, 256, (3,3), 1, 1, 'SELU'], ['M', (2,2), 2, 0]],

    '8': [['C', 1, 128, (3,3), 1, 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 2, 2, 'ReLU'], ['C', 256, 256, (3,3), 1, 3, 3, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0], ['C', 512, 128, (1,1), 1, 1, 0, 'ReLU'], ['C', 128, 128, (3,3), 1, 2, 2, 'ReLU'], ['C', 128, 256, (3,3), 1, 2, 2, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0], ['C', 512, 128, (1,1), 1, 1, 0, 'ReLU'], ['C', 128, 128, (3,3), 1, 3, 3, 'ReLU'], ['C', 128, 256, (3,3), 1, 3, 3, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 1, 'ReLU'],
    ['M', (2,2), 2, 0], ['C', 512, 128, (1,1), 1, 1, 0,'ReLU'], ['C', 128, 128, (3,3), 1, 2, 2, 'ReLU'], ['C', 128, 256, (3,3), 1, 3, 3, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0]],

    '9': [['C', 1, 32, (3,3), 1, 1, 1, 'ReLU'], ['C', 32, 64, (3,3), 1, 2, 2, 'ReLU'], ['C', 64, 128, (3,3), 1, 3, 3, 'ReLU'], ['C', 128, 224, (3,3), 1, 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0], ['C', 224, 32, (1,1), 1, 1, 0, 'ReLU'], ['C', 32, 64, (3,3), 1, 2, 2, 'ReLU'], ['C', 64, 128, (3,3), 1, 2, 2, 'ReLU'], ['C', 128, 224, (3,3), 1, 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0], ['C', 224, 32, (1,1), 1, 1, 0, 'ReLU'], ['C', 32, 64, (3,3), 1, 3, 3, 'ReLU'], ['C', 64, 128, (3,3), 1, 3, 3, 'ReLU'], ['C', 128, 224, (3,3), 1, 1, 1, 'ReLU'],
    ['M', (2,2), 2, 0], ['C', 224, 32, (1,1), 1, 1, 0,'ReLU'], ['C', 32, 64, (3,3), 1, 2, 2, 'ReLU'], ['C', 64, 128, (3,3), 1, 3, 3, 'ReLU'], ['C', 128, 224, (3,3), 1, 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0]],

    '10': [['C', 1, 32, (3,3), 1, 1, 1, 'ReLU'], ['C', 32, 64, (3,3), 1, 2, 2, 'ReLU'], ['C', 64, 128, (3,3), 1, 3, 3, 'ReLU'], ['C', 128, 224, (3,3), 1, 1, 1, 'ReLU'], 
    ['A', (2,2), 2, 0], ['C', 224, 32, (1,1), 1, 1, 0, 'ReLU'], ['C', 32, 64, (3,3), 1, 2, 2, 'ReLU'], ['C', 64, 128, (3,3), 1, 2, 2, 'ReLU'], ['C', 128, 224, (3,3), 1, 1, 1, 'ReLU'], 
    ['A', (2,2), 2, 0], ['C', 224, 32, (1,1), 1, 1, 0, 'ReLU'], ['C', 32, 64, (3,3), 1, 3, 3, 'ReLU'], ['C', 64, 128, (3,3), 1, 3, 3, 'ReLU'], ['C', 128, 224, (3,3), 1, 1, 1, 'ReLU'],
    ['A', (2,2), 2, 0], ['C', 224, 32, (1,1), 1, 1, 0,'ReLU'], ['C', 32, 64, (3,3), 1, 2, 2, 'ReLU'], ['C', 64, 128, (3,3), 1, 3, 3, 'ReLU'], ['C', 128, 224, (3,3), 1, 1, 1, 'ReLU'], 
    ['A', (2,2), 2, 0]],

    '11': [['C', 1, 32, (3,3), 1, 1, 1, 'ReLU'], ['C', 32, 64, (3,3), 1, 1, 1, 'ReLU'], ['C', 64, 128, (5,5), 1, 3, 6, 'ReLU'], ['C', 128, 224, (3,3), 1, 1, 1, 'ReLU'], 
    ['A', (2,2), 2, 0], ['C', 224, 32, (1,1), 1, 1, 0, 'ReLU'], ['C', 32, 64, (3,3), 1, 1, 1, 'ReLU'], ['C', 64, 128, (5,5), 1, 3, 6, 'ReLU'], ['C', 128, 224, (3,3), 1, 1, 1, 'ReLU'], 
    ['A', (2,2), 2, 0], ['C', 224, 32, (1,1), 1, 1, 0, 'ReLU'], ['C', 32, 64, (3,3), 1, 1, 1, 'ReLU'], ['C', 64, 128, (7,7), 1, 3, 9, 'ReLU'], ['C', 128, 224, (3,3), 1, 1, 1, 'ReLU'],
    ['A', (2,2), 2, 0], ['C', 224, 32, (1,1), 1, 1, 0,'ReLU'], ['C', 32, 64, (3,3), 1, 1, 1, 'ReLU'], ['C', 64, 128, (7,7), 1, 3, 9, 'ReLU'], ['C', 128, 224, (3,3), 1, 1, 1, 'ReLU'], 
    ['A', (2,2), 2, 0]],

    '12': [['C', 1, 32, (3,3), 1, 1, 1, 'ReLU'], ['C', 32, 64, (3,3), 1, 1, 1, 'ReLU'], ['C', 64, 128, (5,5), 1, 3, 6, 'ReLU'], ['C', 128, 224, (3,3), 1, 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0], ['C', 224, 32, (1,1), 1, 1, 0, 'ReLU'], ['C', 32, 64, (3,3), 1, 1, 1, 'ReLU'], ['C', 64, 128, (5,5), 1, 3, 6, 'ReLU'], ['C', 128, 224, (3,3), 1, 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0], ['C', 224, 32, (1,1), 1, 1, 0, 'ReLU'], ['C', 32, 64, (3,3), 1, 1, 1, 'ReLU'], ['C', 64, 128, (7,7), 1, 3, 9, 'ReLU'], ['C', 128, 224, (3,3), 1, 1, 1, 'ReLU'],
    ['M', (2,2), 2, 0], ['C', 224, 32, (1,1), 1, 1, 0,'ReLU'], ['C', 32, 64, (3,3), 1, 1, 1, 'ReLU'], ['C', 64, 128, (7,7), 1, 3, 9, 'ReLU'], ['C', 128, 224, (3,3), 1, 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0], ['D2d', .3]],

    '13': [['C', 1, 32, (3,3), 1, 1, 'ReLU'], ['C', 32, 64, (3,3), 1, 1, 'ReLU'], ['C', 64, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'],
    ['M', (2,2), 2, 0], ['C', 256, 32, (1,1), 1, 0, 'ReLU'], ['C', 32, 64, (3,3), 1, 1, 'ReLU'], ['C', 64, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'],
    ['M', (2,2), 2, 0], ['C', 256, 32, (1,1), 1, 0, 'ReLU'], ['C', 32, 64, (3,3), 1, 1, 'ReLU'], ['C', 64, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'],
    ['M', (2,2), 2, 0], ['C', 256, 32, (1,1), 1, 0, 'ReLU'], ['C', 32, 64, (3,3), 1, 1, 'ReLU'], ['C', 64, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'],
    ['M', (2,2), 2, 0], ['C', 256, 32, (1,1), 1, 0, 'ReLU'], ['C', 32, 64, (3,3), 1, 1, 'ReLU'], ['C', 64, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'],
    ['A', (15,15), 2, 0]],

    '14': [['C', 1, 128, (3,3), 1, 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 1, 'ReLU'], ['C', 256, 256, (3,3), 1, 1, 1, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0], ['C', 512, 128, (1,1), 1, 1, 0, 'ReLU'], ['C', 128, 128, (3,3), 1, 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 1, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0], ['C', 512, 128, (1,1), 1, 1, 0, 'ReLU'], ['C', 128, 128, (3,3), 1, 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 1, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0], ['C', 512, 128, (1,1), 1, 1, 0, 'ReLU'], ['C', 128, 128, (3,3), 1, 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 1, 'ReLU'], ['C', 256, 512, (3,3), 1, 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0]],
}

classifier_layers = {
    '1': [['L', 256 * 40 * 30, 1024, 'ReLU'], ['D', .5], ['FC', 1024, 5]],
    '2': [['L', 512 * 7 * 7, 1024, 'ReLU'], ['D', .5], ['L', 1024, 2048, 'ReLU'], ['D', .5], ['FC', 2048, 10]],
    '2.5': [['L', 512 * 15 * 15, 2048, 'ReLU'], ['D', .5], ['FC', 2048, 10]],
    '3': [['L', 512 * 7 * 7, 2048, 'ReLU'], ['D', .5], ['L', 2048, 4096, 'ReLU'], ['D', .5], ['FC', 4096, 10]],
    '4': [['L', 512 * 7 * 7, 1024, 'ReLU'], ['D', .5], ['L', 1024, 2048, 'ReLU'], ['D', .5], ['FC', 2048, 10]],
    '5': [['L', 128 * 6 * 6, 1024, 'ReLU'], ['D', .5], ['L', 1024, 2048, 'ReLU'], ['D', .5], ['FC', 2048, 10]],
    '6': [['L', 256 * 6 * 6, 2048, 'ReLU'], ['D', .5], ['L', 2048, 3192, 'ReLU'], ['D', .5], ['FC', 3192, 10]],
    '7': [['L', 256 * 7 * 7, 1024, 'SELU'], ['AD', .5], ['L', 1024, 2048, 'SELU'], ['AD', .5], ['FC', 2048, 10]],
    '8': [['L', 512 * 15 * 15, 2048, 'ReLU'], ['D', .3], ['FC', 2048, 10]],
    '9': [['L', 224 * 15 * 15, 2048, 'ReLU'], ['D', .3], ['FC', 2048, 10]],
    '9.5': [['L', 224 * 15 * 15, 2048, 'ReLU'], ['D', .7], ['FC', 2048, 10]],
    '10': [['L', 224 * 15 * 15, 2048, 'ReLU'], ['D', .3], ['L', 2048, 3192, 'ReLU'], ['D', .1], ['FC', 3192, 10]],
    '11': [['L', 224 * 15 * 15, 1024, 'ReLU'], ['D', .6], ['L', 1024, 2048, 'ReLU'], ['D', .4], ['FC', 2048, 10]],
    '11.5': [['L', 224 * 15 * 15, 1024, 'ReLU'], ['D', .4], ['FC', 1024, 10]],
    '12': [['L', 224 * 15 * 15, 4096, 'ReLU'], ['D', .4], ['FC', 4096, 10]],
    '13': [['L', 256 * 1 * 1, 2048, 'ReLU'], ['D', .3], ['FC', 2048, 10]],
}


