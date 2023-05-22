from keras import backend
from keras.engine import training
from keras.layers import VersionAwareLayers

layers = None


def ResNet(input_shape, stack_fn, pooling='avg'):
    global layers
    layers = VersionAwareLayers()

    input_tensor = layers.Input(shape=input_shape)

    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    x = layers.Conv2D(64, 3, strides=2, padding='SAME', name="conv1_conv")(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="conv1_bn")(x)
    x = layers.Activation("relu", name="conv1_relu")(x)
    x = stack_fn(x)

    if pooling == "avg":
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    elif pooling == "max":
        x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    # Create model.
    model = training.Model(input_tensor, x)

    return model


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(filters, 1, strides=stride, name=name + "_0_conv")(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn")(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='SAME', name=name + "_1_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(x)
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    x = layers.Conv2D(filters, kernel_size, strides=1, padding="SAME", name=name + "_2_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_2_bn")(x)

    x = layers.Add(name=name + "_add")([shortcut, x])
    x = layers.Activation("relu", name=name + "_out")(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.

    Returns:
      Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, name=name + "_block1")
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + "_block" + str(i))
    return x


def ResNet_new(input_shape, pooling='avg'):
    def stack_fn(x):
        x = stack1(x, 64, 2, name="conv2")
        # x = stack1(x, 128, 2, name="conv3")
        return x

    return ResNet(input_shape, stack_fn, pooling)


if __name__ == "__main__":
    model = ResNet_new((256, 256, 1))
    print(model.summary())
