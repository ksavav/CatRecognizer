import numpy as np
import tensorflow as tf
import keras
import CatRecognizer.Accessories.plot as plt
import Architectures.Xception as xc
import Architectures.AlexNet as alex

# const values
batch_size = 32
img_height = 299
img_width = 299

# path to dataset
path = 'E:/Desktop/PGa/Semestr_9/PUG/Datasets/Cats/test'


# splitting dataset into training, validation and test set
train_ds = keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

data_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal",
                                input_shape=(img_height,
                                             img_width,
                                             3)),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1)
    ]
)

# AlexNet
# alexnet = alex.AlexNet(train_ds.element_spec[0].shape)
# model = alexnet.get_model()

# Xception
xception = xc.Xception(img_height, img_width)
model = xception.get_model()

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# training
epochs = 25
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# plots
plot = plt.Plot(history, epochs)

plot.plot_acc()
plot.plot_loss()

# validate the model
# predictions = model.predict(img_array[0])
# score = tf.nn.softmax(predictions[0])
#
# print(
#     "Should be russian blue -> prediction = {}  {:.2f}%"
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
#
# for i in range(len(class_names)):
#     print(str(class_names[i]) + ' ' + str(round(100 * score[i].numpy(), 2)) + '%')
#
# predictions = model.predict(img_array[1])
# score = tf.nn.softmax(predictions[0])
#
# print(
#     "Should be Egyptian Mau -> prediction = {}  {:.2f}%"
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
#
# for i in range(len(class_names)):
#     print(str(class_names[i]) + ' ' + str(round(100 * score[i].numpy(), 2)) + '%')
