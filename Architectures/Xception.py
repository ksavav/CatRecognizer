from keras.layers import Input, Dense, Conv2D, Add
from keras.layers import SeparableConv2D, ReLU
from keras.layers import BatchNormalization, MaxPool2D
from keras.layers import GlobalAvgPool2D
from keras import Model


class Xception:
    def __init__(self, img_height, img_width, filters_num):
        input = Input(shape=(img_height, img_width, 3))
        self.filters = filters_num
        x = self.entry_flow(input)
        x = self.middle_flow(x)
        output = self.exit_flow(x)

        self.model = Model(inputs=input, outputs=output)

    def get_model(self):
        return self.model

    # creating the Conv-Batch Norm block
    def conv_bn(self, x, filters, kernel_size, strides=1):
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   use_bias=False)(x)
        x = BatchNormalization()(x)
        return x

    # creating separableConv-Batch Norm block
    def sep_bn(self, x, filters, kernel_size, strides=1):
        x = SeparableConv2D(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same',
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        return x

    # entry flow
    def entry_flow(self, x):
        x = self.conv_bn(x, filters=32, kernel_size=3, strides=2)
        x = ReLU()(x)
        x = self.conv_bn(x, filters=64, kernel_size=3, strides=1)
        tensor = ReLU()(x)

        x = self.sep_bn(tensor, filters=128, kernel_size=3)
        x = ReLU()(x)
        x = self.sep_bn(x, filters=128, kernel_size=3)
        x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        tensor = self.conv_bn(tensor, filters=128, kernel_size=1, strides=2)
        x = Add()([tensor, x])

        x = ReLU()(x)
        x = self.sep_bn(x, filters=256, kernel_size=3)
        x = ReLU()(x)
        x = self.sep_bn(x, filters=256, kernel_size=3)
        x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        tensor = self.conv_bn(tensor, filters=256, kernel_size=1, strides=2)
        x = Add()([tensor, x])

        x = ReLU()(x)
        x = self.sep_bn(x, filters=self.filters, kernel_size=3)
        x = ReLU()(x)
        x = self.sep_bn(x, filters=self.filters, kernel_size=3)
        x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        tensor = self.conv_bn(tensor, filters=self.filters, kernel_size=1, strides=2)
        x = Add()([tensor, x])
        return x

    # middle flow
    def middle_flow(self, tensor):
        for _ in range(8):
            x = ReLU()(tensor)
            x = self.sep_bn(x, filters=self.filters, kernel_size=3)
            x = ReLU()(x)
            x = self.sep_bn(x, filters=self.filters, kernel_size=3)
            x = ReLU()(x)
            x = self.sep_bn(x, filters=self.filters, kernel_size=3)
            x = ReLU()(x)
            tensor = Add()([tensor, x])

        return tensor

    # exit flow
    def exit_flow(self, tensor):
        x = ReLU()(tensor)
        x = self.sep_bn(x, filters=self.filters, kernel_size=3)
        x = ReLU()(x)
        x = self.sep_bn(x, filters=1024, kernel_size=3)
        x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        tensor = self.conv_bn(tensor, filters=1024, kernel_size=1, strides=2)
        x = Add()([tensor, x])

        x = self.sep_bn(x, filters=1536, kernel_size=3)
        x = ReLU()(x)
        x = self.sep_bn(x, filters=2048, kernel_size=3)
        x = GlobalAvgPool2D()(x)

        x = Dense(units=1000, activation='softmax')(x)

        return x
