import keras
from keras.datasets import fashion_mnist
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import seaborn as sns

(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()
tt = test_X
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255
test_X = test_X / 255
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

optimizers = ['adam', 'adagrad', 'adadelta', 'adamax', 'nadam']
#
buf_loss = 100
buf_optimizer = 'sgd'
for o in optimizers:
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(10))
    #
    model.add(Activation('softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=o, metrics=['accuracy'])
    model.fit(train_X, train_Y_one_hot, batch_size=64, epochs=1)
    test_loss, test_acc = model.evaluate(test_X, test_Y_one_hot)
    print('Test loss', test_loss)
    print('Test accuracy', test_acc)
    if test_loss < buf_loss:
        buf_loss = test_loss
        buf_optimizer = o

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(10))
#
model.add(Activation('softmax'))

print('Best optimizer is {} with loss {}'.format(buf_optimizer, buf_loss))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=buf_optimizer, metrics=['accuracy'])
model.fit(train_X, train_Y_one_hot, batch_size=64, epochs=1)
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
predictions = model.predict(test_X)
y_true = []
for i in range(10000):
    y_true.append(np.argmax(predictions[i]))
conf_matrix = tf.math.confusion_matrix(labels=test_Y, predictions=y_true).numpy()
conf_matrix_norm = np.around(conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
conf_matrix_df = pd.DataFrame(conf_matrix_norm,
                              index=classes,
                              columns=classes)
figure = plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix_df, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
plt.clf()
for i in range(10000):
    if y_true[i] != test_Y[i]:
        plt.imshow(tt[i])
        plt.text(1, 1, s=str(test_Y[i])+" | "+str(y_true[i]))
        plt.show()
        plt.clf()