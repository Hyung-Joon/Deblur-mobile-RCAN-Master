# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, Lambda, GlobalAveragePooling2D, Multiply, Reshape, \
    LeakyReLU, Dense, Conv2DTranspose
from keras import backend as K
from tensorflow.keras.models import Model


def unnormalize(x):
    x = 255 * (x)
    x = x.round().clip(min=0, max=255)
    x = x.astype(np.uint8)
    return x


def sub_pixel_conv2d(scale=2, **kwargs):
    return Lambda(lambda x: tf.compat.v1.depth_to_space(x, scale), **kwargs)


def ca(input_tensor, filters):

    GAP = GlobalAveragePooling2D()
    Reshape1 = Reshape((1, 1, filters))
    Dense1 = Dense(filters / 4, activation='relu', kernel_initializer='he_normal', use_bias=False)
    Dense2 = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
    Multiply1 = Multiply()

    x = GAP(input_tensor)
    x = Reshape1(x)
    x = Dense1(x)
    x = Dense2(x)
    x = Multiply1([x,input_tensor])

    return x



def rb(input_tensor, filters):
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = ca(x, filters)
    x = Add()([x, input_tensor])

    return x


def ab(input_tensor, filters):
    x = rb(input_tensor, filters)
    x = rb(x, filters)
    concat1 = x

    x = Conv2D(filters * 2, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = rb(x, filters * 2)
    x = rb(x, filters * 2)
    concat2 = x

    x = Conv2D(filters * 4, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    concat3 = x

    x = rb(x, filters * 4)
    x = rb(x, filters * 4)

    x = keras.layers.concatenate([concat3, x], axis=3)  # 192 + 192 = 384

    x = Conv2DTranspose(filters * 2, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = keras.layers.concatenate([concat2, x], axis=3)
    x = rb(x, filters * 4)
    x = rb(x, filters * 4)

    x = Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = keras.layers.concatenate([concat1, x], axis=3)
    x = rb(x, filters * 2)
    x = rb(x, filters * 2)

    x = Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Add()([x, input_tensor])

    return x


def gab(input_tensor, filters):
    x = rb(input_tensor, filters)
    x = rb(x, filters)

    skip_conn = x
    att = x

    att = ab(att, filters)
    att = Conv2D(filters, kernel_size=1, strides=1, padding='same')(att)
    att = Conv2D(filters, kernel_size=3, strides=1, padding='same')(att)
    att = Add()([att, skip_conn])

    x = rb(x, filters)
    concat1 = x
    x = rb(x, filters)
    x = keras.layers.concatenate([concat1, x], axis=3)
    x = Conv2D(filters, kernel_size=1, strides=1, padding='same')(x)
    x = Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
    x = Add()([x, skip_conn])
    x = keras.layers.concatenate([att, x], axis=3)

    x = rb(x, filters * 2)
    x = rb(x, filters * 2)
    x = Conv2D(filters, kernel_size=1, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Add()([x, input_tensor])

    return x


def SN(x, filters=16,reuse=False):
    with tf.variable_scope("sn",reuse=reuse) as scope:
        if reuse : scope.reuse_variables()

    # 32 / 4
        #inputs = Input(shape=(height, width, 3))

        skip_conn = x

        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)

        skip_conn1 = x

        GA = 2
        x = gab(x, filters=filters)
        x1 = x

        for i in range(1, GA):
            x = gab(x, filters=filters)
            x1 = keras.layers.concatenate([x, x1], axis=3)

        x = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

        x = Add()([skip_conn1, x])

        x = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(x)

        x = Add()([skip_conn, x])

        return x #Model(inputs=inputs, outputs=x)




def wsnet(x):


    y= SN(x,reuse=True)
    z= SN(y,reuse=True)

    return z

def wsnet2(height,width):

    inputs = Input(shape=(height, width, 3))
    x = wsnet(inputs)

    return Model(inputs=inputs, outputs=x)




if __name__ == '__main__':
    patch_size = 96
    m = Baseline(patch_size, patch_size)
    m.build(input_shape=(None, patch_size, patch_size, 3))
    os.makedirs('models', exist_ok=True)
    m.save_weights(path.join('models', 'dummy_deblur.hdf5'))

