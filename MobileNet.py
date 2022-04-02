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

tf.test.gpu_device_name()

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
testPath = directory + "/test/images/"
for fileName in os.listdir(testPath):
    imagePath = testPath + fileName
    image = Image.open(imagePath).convert("RGB")
    imageData = np.asarray(image)
    x_test.append(imageData)

x_test = np.array(x_test)
y_test = np.zeros(x_test.size)
x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=0.1,
                                                  random_state=777)
print(x_test.shape)
print(x_train.shape)
print(y_train.shape)
###### Normalizing. ######
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
feature_extractor_model = mobilenet_v2

feature_extractor_layer = hub.KerasLayer(feature_extractor_model,
                                         input_shape=(224, 224, 3),
                                         trainable=False)

model = tf.keras.Sequential([
    keras.layers.Resizing(224, 224), 
    feature_extractor_layer,
    tf.keras.layers.Dense(100, activation = 'softmax')
])

model.build(input_shape=(None, 64, 64, 3))
model.summary()

model.compile(
    #optimizer=tf.keras.optimizers.Adam(),
    optimizer=SGD(learning_rate=0.003, momentum=0.9),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc'])

NUM_EPOCHS = 10

history = model.fit(x_train,
                    y_train,
                    validation_data=(x_val, y_val),
                    epochs=NUM_EPOCHS,
                    batch_size=16)

plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.1, 0.6])
plt.legend(loc='lower right')
plt.show()
