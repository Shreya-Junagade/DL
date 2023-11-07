# Assignment No 6:

Object detection using Transfer Learning of CNN architectures
a. Load in a pre-trained CNN model trained on a large dataset
b. Freeze parameters (weights) in model’s lower convolutional layers
c. Add custom classifier with several layers of trainable parameters to model
d. Train classifier layers on training data available for task
e. Fine-tune hyper parameters and unfreeze more layers as needed

# Object-Recognition-Using-CNN

from keras.datasets import cifar10
import tensorflow as tf
from tensorflow.keras.utils import to_categorical  # Change import to tensorflow.keras.utils
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('Training images: {}'.format(X_train.shape))
print('Testing images: {}'.format(X_test.shape))

print(X_train[0].shape)  # should be and is 32x32x3

# create a grid of 3x3 images (9 images of 3x3 subplots)
for i in range(0, 9):
    plt.subplot(330 + 1 + i)  # denotes 3x3 and postion
    img = X_train[i + 50]  # no need to transpose else transpose([1,2,0])
    plt.imshow(img)

plt.show()

# On the given set, images are blurry (32x32 pixels only), Humans were only 94% accurate in classifying

# Building a convolutional neural network for object recognition on CIFAR-10

print(X_train[0])

seed = 6
np.random.seed(seed)

# again load the dataset as we set the random seed and not applying any shuffling effects or random effects
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize the inputs from 0-255 (RGB) to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

print(X_train[0])

print(y_train.shape)
print(y_train[0])
print(y_train.min())
print(y_train.max())

# hot encode outputs
Y_train = to_categorical(y_train, num_classes=10)  # Change to to_categorical and specify num_classes
Y_test = to_categorical(y_test, num_classes=10)  # Change to to_categorical and specify num_classes

print(Y_train.shape)
print(Y_train[0])

"""# Building the All-CNN"""

from keras.models import Sequential
from keras.layers import Dropout,Activation,Conv2D,GlobalAveragePooling2D
from keras.optimizers import SGD #stochastic gradient descent

def allcnn(weights=None):
    # taking random weights ny default else usr passed pretrained weights
    model = Sequential()  # we will be adding one layer after another

    # not the input layer but need to tell the conv. layer to accept input
    model.add(Conv2D(96,(3,3),padding='same',input_shape=(32,32,3)))#32x32x3 channels
    model.add(Activation('relu'))  # required for each conv. layer
    model.add(Conv2D(96,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96,(3,3),padding='same',strides=(2,2)))
    model.add(Dropout(0.5))  # drop neurons randomly;helps the network generalize(prevent overfitting on training data) better so instead of having individual neurons
    # that are controlling specific classes/features, the features are spread out over the entire network

    model.add(Conv2D(192,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192,(3,3),padding='same',strides=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(192,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192,(1,1),padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10,(1,1),padding='valid'))

    # add GlobalAveragePooling2D layer with Softmax activation
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    # load the weights,if passed
    if weights:
        model.load_weights(weights)

    # return model
    return model



# Define hyperparameters
learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9

# Define training parameters
epochs = 350
batch_size = 32

# Create the SGD optimizer using TensorFlow's optimizer with legacy mode
from tensorflow.keras.optimizers import SGD
sgd = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True, clipnorm=1.0)

model = allcnn()
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

print(model.summary()) # 1.3m parameters and all are trainable

# #fit the model(update the parameters and loss)
model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=epochs,batch_size=batch_size,verbose=1)

# define hyper parameters
learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9
#no need for training parameters

# define weights and build model
weights='all_cnn_weights_0.9088_0.4994.hdf5'#KERAS format hdf5
#pretrained weights  that have already gone through the above press

model=allcnn(weights)

# define optimizer and compile model
sgd = SGD(lr=learning_rate, decay=weight_decay, momentum=momentum, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#measure of model's perfrmane is accuracy

# print model summary
print (model.summary())

# test the model with pretrained weights
scores=model.evaluate(X_test,Y_test,verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

classes=range(0,10)#10 not included

names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# zip the names and classes to make a dictionary of class_labels
class_labels=dict(zip(classes,names))
print(class_labels)
# generate batch of 9 images to predict
batch=X_test[100:109]
labels=np.argmax(Y_test[100:109],axis=-1)

#make predictions
predictions=model.predict(batch,verbose=1)

print(predictions)
print(predictions.shape)

#these are individual class probabilities, should sum to 1.0
for image in predictions:
    print(np.sum(image))

#shows that there is hundred percent probability that images to belong to one of the classes

# use np.argmax() to convert class probabilities to class labels
class_result=np.argmax(predictions,axis=-1)
print(class_result)

#create a grid of 3x3 images
fig,axs=plt.subplots(3,3,figsize=(15,6))
fig.subplots_adjust(hspace=1)
axs=axs.flatten()

for i,img in enumerate (batch):
        # determine label for each prediction, set title
        for key,value in class_labels.items():
            if class_result[i]==key:
                title = 'Prediction: {}\nActual: {}'.format(class_labels[key], class_labels[labels[i]])
                axs[i].set_title(title)
                axs[i].axes.get_xaxis().set_visible(False)
                axs[i].axes.get_yaxis().set_visible(False)

        # plot the image
        axs[i].imshow(img)

# show the plot
plt.show()
