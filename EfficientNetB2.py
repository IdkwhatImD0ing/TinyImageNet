import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.layers.advanced_activations import PReLU
import numpy as np
from PIL import Image
import os
from tensorflow.keras import mixed_precision
import csv

tf.test.gpu_device_name()
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

### LABELS
label_dict = {}
for i, line in enumerate(open("tiny-imagenet-100/wnids.txt", "r")):
    label_dict[line.rstrip("\n")] = i

# Image rescaling
resize = keras.Sequential([keras.layers.Resizing(224, 224)])

### PARSING TRAIN/VALDIATION FILES
directory = "tiny-imagenet-100"
x_train = []
y_train = []
for i, line in enumerate(open("tiny-imagenet-100/wnids.txt", "r")):
    newDirectory = directory + "/train/" + line.rstrip(
        "\n")  # Path for each class in training
    #print("newDirectory")
    #print(newDirectory)
    boxFile = newDirectory + "/" + line.rstrip(
        "\n") + "_boxes.txt"  # Path for the boxes file
    #print("boxFile")
    #print(boxFile)
    for newLine in open(boxFile, "r"):
        fileName = newLine.split('\t')[0]

        imagePath = newDirectory + "/images/" + fileName
        image = Image.open(imagePath).convert("RGB")
        imageData = np.asarray(
            image)  # Image data in numpy array in form (64,64,3)
        #print(imageData.shape)
        #print(imageData)

        x_train.append(imageData)
        y_train.append([label_dict.get(line.rstrip("\n"))])

    #print(fileNames)
print("Finished train")
x_train = np.array(x_train)
y_train = np.array(y_train)

### PARSING TEST FILES
x_test = []
img_id = []
testPath = directory + "/test/images/"
for fileName in os.listdir(testPath):
    imagePath = testPath + fileName
    image = Image.open(imagePath).convert("RGB")
    imageData = np.asarray(image)
    x_test.append(imageData)
    img_id.append(fileName.replace('.JPEG', ''))

x_test = np.array(x_test)
y_test = np.zeros(x_test.size)
x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=0.1,
                                                  random_state=777)
print(x_train.shape)
print(y_train.shape)
###### Normalizing. ######
x_train = x_train
x_val = x_val
x_test = x_test

resModel = keras.applications.EfficientNetB2(weights='imagenet',
                                             include_top=False,
                                             input_shape=(256, 256, 3))

model = keras.models.Sequential()
model.add(keras.layers.UpSampling2D((2, 2)))
model.add(keras.layers.UpSampling2D((2, 2)))
model.add(resModel)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(100, activation='softmax'))

model.build(input_shape=(None, 64, 64, 3))
model.summary()

model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=2e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

NUM_EPOCHS = 20

history = model.fit(x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    epochs=NUM_EPOCHS,
                    batch_size=32)

predictions = model.predict(x_test, batch_size=32)
y_test = np.argmax(predictions, axis=1)
model.evaluate(x_val, y_val)
#print(img_id)
#print(y_test)
combined = [[i, j] for i, j in zip(img_id, y_test)]
with open("predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows([["image_id", "label"]])
    writer.writerows(combined)
model.save("EfficientNetB2")

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.6, 1.])
plt.legend(loc='lower right')
plt.show()