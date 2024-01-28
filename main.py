import numpy as np
import tensorflow as tf
import CatRecognizer.Accessories.plot as plt
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.metrics import Accuracy, SparseCategoricalAccuracy, Precision, Recall
import Architectures.Xception as xc
import Architectures.AlexNet as alex

# const values
batch_size = 32
img_height = 224
img_width = 224
logs_name = 'final_'

# path to dataset

final_final = ['test', 'test_watermark_20%_transparent', 'test_watermark_20%_no_transparent',
               'test_watermark_50%_no_transparent', 'test_watermark_100%_no_transparent']

path = 'E:/Desktop/PGa/Semestr_9/PUG/Datasets/Cats/'

final_results = []


def default_model():
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
        tf.keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])
    model.compile(optimizer=hparams[HP_OPTIMIZER],
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # model.summary()
    return model


def get_model(filters_num, dropout_rate, hparams, model_type, shape=None):
    if model_type == 1:
        # AlexNet
        alexnet = alex.AlexNet(shape)
        return alexnet.get_model()
    elif model_type == 2:
        # Xception
        xception = xc.Xception(img_height, img_width, filters_num, dropout_rate)
        model = xception.get_model()
        model.compile(optimizer='Adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy']) # Accuracy(), SparseCategoricalAccuracy(), Precision(), Recall()])
        return model
    else:
        return default_model()


def train_model(model, num_epochs, hparams, logdir, checkpoints):
    history = model.fit(train_ds, validation_data=val_ds, epochs=num_epochs,
                        callbacks=[
                            tf.keras.callbacks.TensorBoard(logdir),  # log metrics
                            hp.KerasCallback(logdir, hparams),  # log hparams
                            checkpoints,  # save
                            lr_callback
                        ]
                        )
    accuracy = history.history['accuracy']

    # _, accuracy = model.evaluate(x_test, y_test)
    return history, accuracy


# defined learning rate
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=4,
    min_lr=0.0001
)

# check if the gpu is available
tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# metric
METRIC_ACCURACY = 'accuracy'

# hparams
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32, 64]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.4]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
HP_FILTERS = hp.HParam('filters', hp.Discrete([728, 768]))

for ds in final_final:
    temp_logs_name = logs_name + ds

    with tf.summary.create_file_writer(f'logs/final/{temp_logs_name}').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_FILTERS],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        )

    # splitting dataset into training and test set
    train_ds = tf.keras.utils.image_dataset_from_directory(
        path + ds,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        path + ds,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_batches = tf.data.experimental.cardinality(val_ds)

    test_ds = val_ds.take(val_batches // 2)
    val_ds = val_ds.skip(val_batches // 2)

    class_names = train_ds.class_names
    print(class_names)

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    # training
    epochs = 20
    session_num = 0

    for dropout_rate in HP_DROPOUT.domain.values:
        #for filters_num in HP_FILTERS.domain.values:
        #for optimizer in HP_OPTIMIZER.domain.values:
        hparams = {
            HP_DROPOUT: dropout_rate,
            #HP_OPTIMIZER: optimizer,
            #HP_FILTERS: filters_num
        }
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        model = get_model(728, dropout_rate, hparams, model_type=2, shape=train_ds.element_spec[0].shape)
        # model.summary()
        checkpoint = tf.keras.callbacks.ModelCheckpoint(f'/checkpoints/cp-{epochs:02d}.h5', monitor="accuracy", save_best_only=True, mode='max')
        history, accuracy = train_model(model, epochs, hparams, f'logs/final/{temp_logs_name}/' + run_name + "filters" + '728' + "dropout_rate" + str(dropout_rate) + "epo" + str(epochs), checkpoint)
        session_num += 1
        results = model.evaluate(test_ds, batch_size=32)
        final_results.append(results)
        print("test loss, test acc:", results)

# plots
# plot = plt.Plot(history, epochs)
#
# plot.plot_acc()
# plot.plot_loss()

for i in range(len(final_results)):
    print(f'{final_final[i]} - test loss, test acc: {final_results[i]}')

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
