import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, LeakyReLU, Add
from tensorflow.keras import initializers

class ConvLayer(Layer):
    def __init__(self, num_filters, activation):
        super().__init__()
        self.num_filters = num_filters
        self.activation = activation
        self.conv2d = Conv2D(self.num_filters, 3, (1, 1), activation= self.activation, kernel_initializer= initializers.random_normal)

    def call(self, x):
        x = self.conv2d(x)
        return x

def conv_net(x, z):

    x = keras.Input(shape=(x.shape[1], x.shape[2], x.shape[3]))
    z = keras.Input(shape=(z.shape[1], z.shape[2], z.shape[3]))

    i = Add()([x, z])

    conv_layer1 = ConvLayer(32, LeakyReLU)
    conv_layer2 = ConvLayer(32, LeakyReLU)
    conv_layer3 = ConvLayer(32, LeakyReLU)
    conv_layer4 = ConvLayer(32, LeakyReLU)
    conv_layer5 = ConvLayer(3, 'tanh')

    conv1 = conv_layer1(i)
    conv2 = conv_layer2(conv1)
    conv3 = conv_layer3(conv2)
    conv4 = conv_layer4(conv3)
    conv5 = conv_layer5(conv4)

    out = Add()([x, conv5])

    return out


