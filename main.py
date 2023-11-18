import numpy as np
import tensorflow as tf
import keras
import CatRecognizer.Accessories.plot as plt

# const values
batch_size = 32
img_height = 180
img_width = 180

# path to dataset
path = 'C:/Users/axawe/Desktop/Projects/CatsRecognizer/Dataset/Cats/test'

# tests_images = {'russian': 'E:/Desktop/russianblue.jpg',
#                 'egypt': 'E:/Desktop/egypt.jpg'}
#
# img_array = []

# for key in tests_images:
#     img = tf.keras.utils.load_img(
#         tests_images[key], target_size=(img_height, img_width)
#     )
#
#     xd = tf.keras.utils.img_to_array(img)
#     xd = tf.expand_dims(xd, 0)  # Create a batch
#
#     img_array.append(xd)

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

# for image_batch, labels_batch in train_ds:
#   print(image_batch.shape)
#   print(labels_batch.shape)
#   break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

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

# structure of the model layers
model = keras.models.Sequential([
    data_augmentation,
    keras.layers.Rescaling(1. / 255),
    keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# training
epochs = 2
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
