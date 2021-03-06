import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, ReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from pdb import set_trace
from pathlib import Path
path = Path("simulator-data")
df = pd.read_csv(path/"driving_log.csv")
split = int(df.shape[0]*0.85) # separates train and val datasets
IMG_HEIGHT, IMG_WIDTH = 160-(70+25), 320



def preprocess(image, measurement, correction_factor, flip=1.0):
    img, label = np.copy(image), np.copy(measurement)
    # Modifies the target output if selected image is from right or left camera
    label += 0.2 * correction_factor
    # Scale and 0-center pixels (improves gradient handling)
    img = img / 255 - 0.5
    # Flips the image left/right randomly
    if np.random.uniform() < flip:
        img = np.fliplr(img)
        label = -label
    return img, label

def load_image(img):
    image = cv2.cvtColor(cv2.imread(str(path/img)), cv2.COLOR_BGR2RGB)
    # Crop to out region above track and region of the car hood.
    return image[70:-25,...]

def get_batch(rows):
    """ ows are from the dataframe, each containing image filenames and steering angle """
    images, targets = [], []
    for i in range(rows.shape[0]):
        # randomly select left/center/right camera for each data sample
        idx = np.random.choice(3)
        img = load_image(rows[i,idx].strip())
        tgt = rows[i,3] # 'steering'
        # To compensate for the vehicle being biased in understeering on turns,
        # we keep the first 24 images in a batchsize of 64 to ensure minimum batchsize of 24,
        # and after that keep those whose steering angle is greater than .01
        # This reduces the occurence of driving straight through sharp turns
        if len(images) > 24 and np.abs(tgt) < .01:
            continue
        # Adjust target if left/right camera is selected
        if idx == 0: correction_factor = 0
        elif idx == 1: correction_factor = 1
        else: correction_factor = -1
        img, tgt = preprocess(img, tgt, correction_factor, flip=0.5)
        images.append(img)
        targets.append(tgt)
    return np.array(images), np.array(targets)


def train_epoch(model, df, bs=64):
    losses = []
    n_samples = df.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    for i in range(n_samples // bs - 1):
        images, targets = get_batch(df.iloc[indices[bs*i:bs*(i+1)]].values)
        # Image augmentation
        for img in images:
            img = tf.image.adjust_brightness(img, np.random.uniform(0.8, 1.2))
            img = tf.image.adjust_saturation(img, np.random.uniform(0.8, 1.2))
        hist = model.fit(images, targets, verbose=0)
        losses.append(hist.history['loss'])
        #if i % 200 == 0 and i > 0:
        #    print(np.mean(losses))
    return np.mean(losses)


def valid_epoch(model, df, bs=8):
    losses = []
    n_samples = df.shape[0]
    indices = np.arange(n_samples)
    for i in range(n_samples // bs - 1):
        idx = indices[bs*i:bs*(i+1)]
        images = np.array(
                    [(load_image(df.iloc[j,0].strip()) / 255 - 0.5) for j in idx]
                    + [(load_image(df.iloc[j,1].strip()) / 255 - 0.5) for j in idx]
                    + [(load_image(df.iloc[j,2].strip()) / 255 - 0.5) for j in idx]
                )
        targets = np.array([df.iloc[j,3] for j in idx] + [df.iloc[j,3]+.2 for j in idx] + [df.iloc[j,3]-.2 for j in idx])
        outputs = model.predict(images)
        losses.append(np.mean((targets - outputs)**2))
    return np.mean(losses)


# Buid CNN with:
#   5 convolutions layers, each followed by batchnorm, maxpool and dropout (last layer has ReLU instead of maxpool)
#   Flatten the last convolutional layer
#   3 fully connected layers, the first two have dropout
model = Sequential([
    Conv2D(32, 7, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(64, 5, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(128, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(128, 3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(256, 3, padding='same', activation='relu'),
    Flatten(),
    BatchNormalization(),
    ReLU(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    ReLU(),
    Dropout(0.5),
    Dense(32, activation='relu'),
    ReLU(),
    Dropout(0.2),
    Dense(1)
])

# model = tf.keras.models.load_model('model.h5')

# After trying some different learning rates, 1e-5 seemed to perform very well compared to higher learning rates.
# A faster approach would have been to use cyclical learning rates.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='mean_squared_error')


# After each epoch, save model if validation error has decreased.
train_losses = []
val_losses = []
best_val = 1000
for epoch in range(30):
   loss = train_epoch(model, df[:split])
   train_losses.append(loss)
   loss = valid_epoch(model, df[split:])
   val_losses.append(loss)
   print("EPOCH {}: {:.5f}, {:.5f}".format(epoch, train_losses[-1], val_losses[-1]))
   if loss < best_val and epoch > 0:
       best_val = loss
        model.save('model.h5')
