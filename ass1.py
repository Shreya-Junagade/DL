# Study of Deep learning Packages: Tensorflow, Keras, Theano and PyTorch. Document the distinct
# features and functionality of the packages


#import libraries
from keras.datasets import cifar10
from matplotlib import pyplot


# loading
(train_X, train_y), (test_X, test_y) = cifar10.load_data()

# shape of dataset
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test: ' + str(test_X.shape))
print('Y_test: ' + str(test_y.shape))

# plotting
from matplotlib import pyplot

for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()
