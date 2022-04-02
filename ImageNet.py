import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.layers.advanced_activations import PReLU
import numpy as np
from PIL import Image
import os
import csv

tf.test.gpu_device_name()

### LABELS
label_dict = {}
for i, line in enumerate(open("tiny-imagenet-100/wnids.txt", "r")):
    label_dict[line.rstrip("\n")] = i

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
print(x_test.shape)

x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=0.1)
print(x_train.shape)
print(y_train.shape)
###### Normalizing. ######
x_train = x_train / 255
x_val = x_val / 255
#x_test = x_test/255

###### Optimized Neural Network. ######
model = keras.models.Sequential()
# Data Augmentation
#model.add(keras.layers.RandomFlip("horizontal_and_vertical"))
#model.add(keras.layers.RandomRotation(0.2))
#model.add(keras.layers.RandomContrast(0.2))
#model.add(keras.layers.RandomZoom(0.2))

# Model Layers
model.add(
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=PReLU()))
model.add(
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=PReLU()))
model.add(keras.layers.MaxPooling2D((2, 2), strides=2))
model.add(keras.layers.BatchNormalization())
model.add(
    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=PReLU()))
model.add(keras.layers.MaxPooling2D((2, 2), strides=2))
model.add(keras.layers.BatchNormalization())
model.add(
    keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation=PReLU()))
model.add(keras.layers.MaxPooling2D((2, 2), strides=2))
model.add(
    keras.layers.Conv2D(filters=200, kernel_size=(1, 1), activation=PReLU()))
model.add(keras.layers.GlobalAveragePooling2D())
#odel.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(2048, activation = PReLU()))
#model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(100, activation='softmax'))

model.build(input_shape=(None, 64, 64, 3))
model.summary()
model.compile(optimizer=SGD(learning_rate=0.003, momentum=0.9),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
history = model.fit(x_train,
                    y_train,
                    epochs=5,
                    validation_data=(x_val, y_val),
                    batch_size=16)

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

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.1, 0.6])
plt.legend(loc='lower right')
plt.show()
