# USAGE
# python train_mask_detector.py --dataset dataset
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True)
parser.add_argument("-m", "--model", type=str,default="mask.model")
argument = vars(parser.parse_args())

learning_rate = 1e-4
epoch = 20
batch = 32
image_paths = list(paths.list_images(argument["dataset"]))
image_data = []
image_labels = []

for i in image_paths:
	label = i.split(os.path.sep)[-2]
	image = load_img(i, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)
	image_data.append(image)
	image_labels.append(label)

image_data = np.array(image_data, dtype="float32")
image_labels = np.array(image_labels)
lb = LabelBinarizer()
image_labels = lb.fit_transform(image_labels)
image_labels = to_categorical(image_labels)
(trainX, testX, trainY, testY) = train_test_split(image_data, image_labels,test_size=0.15, stratify=image_labels, random_state=42)
aug = ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")
orig_model = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

headModel = orig_model.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
model = Model(inputs=orig_model.input, outputs=headModel)
for layer in orig_model.layers:
	layer.trainable = False 
opt = Adam(lr=learning_rate, decay=learning_rate / epoch)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

M = model.fit(aug.flow(trainX, trainY, batch_size=batch),
	steps_per_epoch=len(trainX) // batch,
	validation_data=(testX, testY),
	validation_steps=len(testX) // batch,
	epochs=epoch)

predIdxs = model.predict(testX, batch_size=batch)
predIdxs = np.argmax(predIdxs, axis=1)
model.save(argument["model"], save_format="h5")
