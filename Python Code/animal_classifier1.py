import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential
import sys
from os import listdir
import os
from os.path import isfile, join
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 as  PretrainedModel,preprocess_input

batch_size = 32
img_height = 200
img_width = 200
AUTOTUNE = tf.data.AUTOTUNE

data_dir = 'animals/animals'

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  labels='inferred',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

validation_dataset = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  labels='inferred',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)


def create_model():
    model = Sequential()

    base_model=tf.keras.applications.xception.Xception(weights='imagenet',include_top=False, input_shape=(200,200,3))
    model.add(base_model)
    model.add(Conv2D(128, (3, 3), activation='relu')),
    model.add(Dropout(.5))
    model.add(Flatten())
    model.add(Dense(90, activation='softmax'))

    return model

model = create_model()

model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


history  = model.fit(train_ds, epochs=25,validation_data=validation_dataset)
model.save('my_model1.keras')
model.save('model1.h5')

