import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


batch_size = 32
img_height = 180
img_width = 180

path = 'E:/Desktop/PGa/Semestr_9/PUG/Datasets/Final_dataset_changed_names_no_labels/cats'

tests_images = {'russian': 'E:/Desktop/russianblue.jpg',
                'egypt': 'E:/Desktop/egypt.jpg'}

img_array = []

for key in tests_images:
    img = tf.keras.utils.load_img(
        tests_images[key], target_size=(img_height, img_width)
    )

    xd = tf.keras.utils.img_to_array(img)
    xd = tf.expand_dims(xd, 0)  # Create a batch

    img_array.append(xd)


train_ds = tf.keras.utils.image_dataset_from_directory(
  path,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
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

data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
  ]
)

model = tf.keras.models.Sequential([
  data_augmentation,
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 20
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

predictions = model.predict(img_array[0])
score = tf.nn.softmax(predictions[0])

print(
    "Should be russian blue -> prediction = {}  {:.2f}%"
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

for i in range(len(class_names)):
    print(str(class_names[i]) + ' ' + str(round(100 * score[i].numpy(), 2)) + '%')

predictions = model.predict(img_array[1])
score = tf.nn.softmax(predictions[0])

print(
    "Should be Egyptian Mau -> prediction = {}  {:.2f}%"
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

for i in range(len(class_names)):
    print(str(class_names[i]) + ' ' + str(round(100 * score[i].numpy(), 2)) + '%')
