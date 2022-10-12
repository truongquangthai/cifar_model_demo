import numpy
import matplotlib.pyplot as pyplot
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical

# with picture x has label y
(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()


classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

xtrain = xtrain/255
xtest = xtest/255

ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)
# model_training = models.Sequential([
#     layers.Conv2D(32,(3,3), input_shape=(32, 32, 3), activation='relu'),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.15),

#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.2),

#     layers.Conv2D(128, (3,3),input_shape=(32, 32, 3), activation='relu'),
#     layers.MaxPool2D((2, 2)),
#     layers.Dropout(0.2),

#     # layers.Conv2D(256, (3,3), input_shape=(32, 32, 3), activation='relu'),
#     # layers.MaxPool2D((2,2)),
#     # layers.Dropout(0.3),

#     layers.Flatten(),
#     layers.Dense(3072, activation='relu'),
#     layers.Dense(1536, activation='relu'),
#     layers.Dense(768, activation='relu'),
#     layers.Dense(384, activation='relu'),
#     layers.Dense(10, activation='softmax'),
# ])
# #model_training.summary()

# model_training.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #Adam is popular currently, in this example we use Stochastic Gradient Descent
# model_training.fit(xtrain,ytrain,epochs=20)

# model_training.save('cifar10_model.h5')

models = models.load_model('cifar10_model.h5')

numpy.random.shuffle(xtest)

for i in range(24):
    pyplot.subplot(8,3, i+1)
    pyplot.imshow(xtest[500+i])
    pyplot.title(classes[numpy.argmax(models.predict(xtest[500 + i].reshape((-1,32,32,3))))])
    pyplot.axis('off')

pyplot.show()