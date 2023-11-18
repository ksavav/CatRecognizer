import matplotlib.pyplot as plt


class Plot:
    def __init__(self, history, epochs):
        self.epochs = range(epochs)
        self.acc = history.history['accuracy']
        self.val_acc = history.history['val_accuracy']
        self.loss = history.history['loss']
        self.val_loss = history.history['val_loss']

    def plot_loss(self):
        plt.figure(figsize=(8, 8))
        plt.plot(self.epochs, self.loss, label='Training Loss')
        plt.plot(self.epochs, self.val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def plot_acc(self):
        plt.figure(figsize=(8, 8))
        plt.plot(self.epochs, self.acc, label='Training Accuracy')
        plt.plot(self.epochs, self.val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')


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