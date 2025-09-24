# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% colab={"base_uri": "https://localhost:8080/"} id="Yuk3ZuBvc5Ic" outputId="bf32a98b-ac9f-4fb3-91d2-92483909b355"
# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub

paultimothymooney_chest_xray_pneumonia_path = kagglehub.dataset_download(
    "paultimothymooney/chest-xray-pneumonia"
)

print("Data source import complete.")


# %% [markdown] id="IUjTdt6Hc5Ic"
# # What is Pneumonia?
# **Pneumonia is an inflammatory condition of the lung affecting primarily the small air sacs known as alveoli.Symptoms typically include some combination of productive or dry cough, chest pain, fever and difficulty breathing. The severity of the condition is variable. Pneumonia is usually caused by infection with viruses or bacteria and less commonly by other microorganisms, certain medications or conditions such as autoimmune diseases.Risk factors include cystic fibrosis, chronic obstructive pulmonary disease (COPD), asthma, diabetes, heart failure, a history of smoking, a poor ability to cough such as following a stroke and a weak immune system. Diagnosis is often based on symptoms and physical examination. Chest X-ray, blood tests, and culture of the sputum may help confirm the diagnosis.The disease may be classified by where it was acquired, such as community- or hospital-acquired or healthcare-associated pneumonia.**
# ![image.png](attachment:image.png)

# %% _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _kg_hide-output=true colab={"base_uri": "https://localhost:8080/"} id="i2JrQFnUc5Id" outputId="54fc35fa-7137-4aca-9c6f-0b76e71e53cd"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import os

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [markdown] id="HnnQ77K0c5Ie"
# # Importing the necessary libraries

import os

import cv2
import keras

# %% _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" id="Rjdixy5hc5Ie"
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import ReduceLROnPlateau
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %% [markdown] id="M3xqAnVAc5If"
# # Description of the Pneumonia Dataset
# **The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).
# Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.
# For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.**

# %% id="qBSfejDXc5If"
labels = ["PNEUMONIA", "NORMAL"]
img_size = 150


def get_training_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(
                    img_arr, (img_size, img_size)
                )  # Reshaping images to preferred size
                if resized_arr.shape == (
                    img_size,
                    img_size,
                ):  # Check if resized successfully
                    data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data, dtype=object)


# %% [markdown] id="rZmUnryHc5Ig"
# # Loading the Dataset

# %% _kg_hide-output=true id="KZeMIGEKc5Ig"
train = get_training_data("/kaggle/input/chest-xray-pneumonia/chest_xray/train")
test = get_training_data("/kaggle/input/chest-xray-pneumonia/chest_xray/test")
val = get_training_data("/kaggle/input/chest-xray-pneumonia/chest_xray/val")

# %% [markdown] id="m-VzyUj2c5Ig"
# # Data Visualization & Preprocessing

# %% colab={"base_uri": "https://localhost:8080/", "height": 466} id="-uhk3y0_c5Ig" outputId="1397fa95-f7ec-4245-c3b4-b6d53b0bf21e"
l = []
for i in train:
    if i[1] == 0:
        l.append("Pneumonia")
    else:
        l.append("Normal")
sns.set_style("darkgrid")
sns.countplot(l)

# %% [markdown] id="jOvaUE9Uc5Ig"
# **The data seems imbalanced . To increase the no. of training examples, we will use data augmentation**

# %% [markdown] id="nRSGGNN3c5Ih"
# **Previewing the images of both the classes**

# %% colab={"base_uri": "https://localhost:8080/", "height": 936} id="BPNybnFQc5Ih" outputId="02ec9553-0beb-4290-a409-36e163488ee2"
plt.figure(figsize=(5, 5))
plt.imshow(train[0][0], cmap="gray")
plt.title(labels[train[0][1]])

plt.figure(figsize=(5, 5))
plt.imshow(train[-1][0], cmap="gray")
plt.title(labels[train[-1][1]])

# %% id="Bt4Sy5aVc5Ih"
x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

# %% [markdown] id="Nnvbc655c5Ih"
# **We perform a grayscale normalization to reduce the effect of illumination's differences.Moreover the CNN converges faster on [0..1] data than on [0..255].**

# %% id="4ogYIydCc5Ih"
# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

# %% id="goQrpznqc5Ih"
# resize data for deep learning
x_train = x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val = x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

x_test = x_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)

# %% [markdown] id="sv_-QBd_c5Ii"
# # Data Augmentation
# **In order to avoid overfitting problem, we need to expand artificially our dataset. We can make your existing dataset even larger. The idea is to alter the training data with small transformations to reproduce the variations.
# Approaches that alter the training data in ways that change the array representation while keeping the label the same are known as data augmentation techniques. Some popular augmentations people use are grayscales, horizontal flips, vertical flips, random crops, color jitters, translations, rotations, and much more.
# By applying just a couple of these transformations to our training data, we can easily double or triple the number of training examples and create a very robust model.**

# %% id="gQt_lYzxc5Ii"
# With data augmentation to prevent overfitting and handling the imbalance in dataset

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.2,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,
)  # randomly flip images


datagen.fit(x_train)

# %% [markdown] id="KVL3HaEEc5Ii"
# For the data augmentation, i choosed to :
# 1. Randomly rotate some training images by 30 degrees
# 2. Randomly Zoom by 20% some training images
# 3. Randomly shift images horizontally by 10% of the width
# 4. Randomly shift images vertically by 10% of the height
# 5. Randomly flip images horizontally.
# Once our model is ready, we fit the training dataset.

# %% [markdown] id="ybJD6eU_c5Ii"
# # Training the Model

# %% colab={"base_uri": "https://localhost:8080/", "height": 968} id="SIBJvHaJc5Ii" outputId="bce20e21-632d-4708-eb0e-cbaa6b648102"
model = Sequential()
model.add(
    Conv2D(
        32,
        (3, 3),
        strides=1,
        padding="same",
        activation="relu",
        input_shape=(150, 150, 1),
    )
)
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Conv2D(256, (3, 3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation="sigmoid"))
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# %% id="aiZDQaiGc5Ij"
learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_accuracy", patience=2, verbose=1, factor=0.3, min_lr=0.000001
)

# %% colab={"base_uri": "https://localhost:8080/"} id="WhBjGRitc5Ij" outputId="842b89d1-f53b-4330-fc97-776751129c55"
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=12,
    validation_data=datagen.flow(x_val, y_val),
    callbacks=[learning_rate_reduction],
)

# %% colab={"base_uri": "https://localhost:8080/"} id="fZ7pMl6Tc5Ij" outputId="def3e476-4f44-40c7-8dd2-0289776b73ee"
print("Loss of the model is - ", model.evaluate(x_test, y_test)[0])
print("Accuracy of the model is - ", model.evaluate(x_test, y_test)[1] * 100, "%")

# %% [markdown] id="sueEtlDMc5Ij"
# # Analysis after Model Training

# %% colab={"base_uri": "https://localhost:8080/", "height": 584} id="FH0lawLnc5Ij" outputId="1bab1cb5-b3f3-43e8-de3b-6c5b0aa47c44"
epochs = [i for i in range(12)]
fig, ax = plt.subplots(1, 2)
train_acc = history.history["accuracy"]
train_loss = history.history["loss"]
val_acc = history.history["val_accuracy"]
val_loss = history.history["val_loss"]
fig.set_size_inches(20, 10)

ax[0].plot(epochs, train_acc, "go-", label="Training Accuracy")
ax[0].plot(epochs, val_acc, "ro-", label="Validation Accuracy")
ax[0].set_title("Training & Validation Accuracy")
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs, train_loss, "g-o", label="Training Loss")
ax[1].plot(epochs, val_loss, "r-o", label="Validation Loss")
ax[1].set_title("Testing Accuracy & Loss")
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="8_tpX02Nc5Ik" outputId="d4244cdf-ad80-4ba5-a9a6-3c3d5c030d63"
predictions = (model.predict(x_test) > 0.5).astype("int32")
predictions = predictions.reshape(1, -1)[0]
predictions[:15]

# %% colab={"base_uri": "https://localhost:8080/"} id="qCAEKWegc5Ik" outputId="26f3ae97-4bc4-442a-cb2f-9402622bbf01"
print(
    classification_report(
        y_test, predictions, target_names=["Pneumonia (Class 0)", "Normal (Class 1)"]
    )
)

# %% colab={"base_uri": "https://localhost:8080/"} id="Yaj8aalCc5Ik" outputId="d4a7bc25-7e09-4c01-d4b0-fcd2bde9068f"
cm = confusion_matrix(y_test, predictions)
cm

# %% id="D4uf0_6yc5Ik"
cm = pd.DataFrame(cm, index=["0", "1"], columns=["0", "1"])

# %% colab={"base_uri": "https://localhost:8080/", "height": 847} id="GjCwWvjGc5Ik" outputId="f3a4a921-7190-4bc9-8c2e-7dc5e0f602f2"
plt.figure(figsize=(10, 10))
sns.heatmap(
    cm,
    cmap="Blues",
    linecolor="black",
    linewidth=1,
    annot=True,
    fmt="",
    xticklabels=labels,
    yticklabels=labels,
)

# %% id="eV5fkBUnc5Il"
correct = np.nonzero(predictions == y_test)[0]
incorrect = np.nonzero(predictions != y_test)[0]

# %% [markdown] id="ePnV4yu-c5Il"
# **Some of the Correctly Predicted Classes**

# %% colab={"base_uri": "https://localhost:8080/", "height": 487} id="OWPqZBe7c5Il" outputId="92819686-aa1b-4098-b08a-43ccf27b4d18"
i = 0
for c in correct[:6]:
    plt.subplot(3, 2, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[c].reshape(150, 150), cmap="gray", interpolation="none")
    plt.title("Predicted Class {},Actual Class {}".format(predictions[c], y_test[c]))
    plt.tight_layout()
    i += 1

# %% [markdown] id="sLhAeOgWc5Il"
# **Some of the Incorrectly Predicted Classes**

# %% colab={"base_uri": "https://localhost:8080/", "height": 487} id="XWDQ7KWkc5Il" outputId="e769f72c-ce2b-431e-c26c-16658bddd89b"
i = 0
for c in incorrect[:6]:
    plt.subplot(3, 2, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[c].reshape(150, 150), cmap="gray", interpolation="none")
    plt.title("Predicted Class {},Actual Class {}".format(predictions[c], y_test[c]))
    plt.tight_layout()
    i += 1

# %% id="7UMKPZrNc5Il"
