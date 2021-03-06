import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2D, Conv3D, Input, AveragePooling2D, \
    multiply, Dense, Dropout, Flatten, AveragePooling3D
from tensorflow.python.keras.models import Model


#%%

class Attention_mask(tf.keras.layers.Layer):
    def call(self, x):
        xsum = K.sum(x, axis=1, keepdims=True)
        xsum = K.sum(xsum, axis=2, keepdims=True)
        xshape = K.int_shape(x)
        return x / xsum * xshape[1] * xshape[2] * 0.5

    def get_config(self):
        config = super(Attention_mask, self).get_config()
        return config

# %%

def DeepPhy(nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3), dropout_rate1=0.25, dropout_rate2=0.5,
            pool_size=(2, 2), nb_dense=128):

    diff_input = Input(shape=input_shape)
    rawf_input = Input(shape=input_shape)

    d1 = Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(diff_input)
    d2 = Conv2D(nb_filters1, kernel_size, activation='tanh')(d1)

    r1 = Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv2D(nb_filters1, kernel_size, activation='tanh')(r1)

    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    gated1 = multiply([d2, g1])
    print(gated1.shape)

    d3 = AveragePooling2D(pool_size)(gated1)
    d4 = Dropout(dropout_rate1)(d3)

    r3 = AveragePooling2D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)

    d5 = Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(d4)
    d6 = Conv2D(nb_filters2, kernel_size, activation='tanh')(d5)

    r5 = Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv2D(nb_filters2, kernel_size, activation='tanh')(r5)

    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    gated2 = multiply([d6, g2])

    d7 = AveragePooling2D(pool_size)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    d9 = Flatten()(d8)
    d10 = Dense(nb_dense, activation='tanh')(d9)
    d11 = Dropout(dropout_rate2)(d10)
    out = Dense(1)(d11)
    model = Model(inputs=[diff_input, rawf_input], outputs=out)
    return model


#%% 2DCNN-MT
def DeepPhys_2DCNN_MT(nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3), dropout_rate1=0.25, dropout_rate2=0.5,
            pool_size=(2, 2), nb_dense=128, use_dataloader=False):

    diff_input = Input(shape=input_shape)
    rawf_input = Input(shape=input_shape)

    d1 = Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(diff_input)
    d2 = Conv2D(nb_filters1, kernel_size, activation='tanh')(d1)

    r1 = Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv2D(nb_filters1, kernel_size, activation='tanh')(r1)

    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    gated1 = multiply([d2, g1])

    d3 = AveragePooling2D(pool_size)(gated1)
    d4 = Dropout(dropout_rate1)(d3)

    r3 = AveragePooling2D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)

    d5 = Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(d4)
    d6 = Conv2D(nb_filters2, kernel_size, activation='tanh')(d5)

    r5 = Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv2D(nb_filters2, kernel_size, activation='tanh')(r5)

    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    gated2 = multiply([d6, g2])

    d7 = AveragePooling2D(pool_size)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    d9 = Flatten()(d8)
    d10_y = Dense(nb_dense, activation='tanh')(d9)
    d11_y = Dropout(dropout_rate2)(d10_y)
    out_y = Dense(1, name='pulse')(d11_y)

    d10_r = Dense(nb_dense, activation='tanh')(d9)
    d11_r = Dropout(dropout_rate2)(d10_r)
    out_r = Dense(1, name='resp')(d11_r)

    model = Model(inputs=[diff_input, rawf_input], outputs=[out_y, out_r])
    return model


#%%

class Attention_Split(tf.keras.layers.Layer):
    def call(self, x):
        averaged_x = K.mean(x, axis=0)
        averaged_x = tf.squeeze(averaged_x)
        x0, x1 = tf.split(averaged_x, 2, axis=0)
        x0_0, x0_1 = tf.split(x0, 2, axis=1)
        x1_0, x1_1 = tf.split(x1, 2, axis=1)
        x0_0 = tf.reduce_mean(x0_0) # 17 x 17
        x0_1 = tf.reduce_mean(x0_1)
        x1_0 = tf.reduce_mean(x1_0)
        x1_1 = tf.reduce_mean(x1_1)
        # all = tf.tensor([x0_0, x0_1, x1_0, x1_1])
        return x0_0, x0_1, x1_0, x1_1

# at_split = Attention_Split()
# test_tensor = tf.random.normal([20, 36, 36, 1])
# a, b, c, d = at_split(test_tensor)
# print('a', a)
# print('b', b)
# print('c', c)
# print('d', d)

#%%

class TSM(tf.keras.layers.Layer):
    def call(self, x, n_frame, fold_div=3):
        nt, h, w, c = x.shape
        x = K.reshape(x, (-1, n_frame, h, w, c))
        fold = c // fold_div
        last_fold = c - (fold_div-1)*fold
        out1, out2, out3 = tf.split(x, [fold, fold, last_fold], axis=-1)

        # Shift left
        padding_1 = tf.zeros_like(out1)
        padding_1 = padding_1[:,-1,:,:,:]
        padding_1 = tf.expand_dims(padding_1, 1)
        _, out1 = tf.split(out1, [1, n_frame-1], axis=1)
        out1 = tf.concat([out1, padding_1], axis=1)

        # Shift right
        padding_2 = tf.zeros_like(out2)
        padding_2 = padding_2[:,0,:,:, :]
        padding_2 = tf.expand_dims(padding_2, 1)
        out2, _ = tf.split(out2, [n_frame-1,1], axis=1)
        out2 = tf.concat([padding_2, out2], axis=1)

        out = tf.concat([out1, out2, out3], axis=-1)
        out = K.reshape(out, (-1, h, w, c))

        return out

    def get_config(self):
        config = super(TSM, self).get_config()
        return config


def TSM_Cov2D(x, n_frame, nb_filters=128, kernel_size=(3, 3), activation='tanh', padding='same'):
    x = TSM()(x, n_frame)
    x = Conv2D(nb_filters, kernel_size, padding=padding, activation=activation)(x)
    return x

#%% CONV2D + TSM

def TS_CAN(n_frame, nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3), dropout_rate1=0.25, dropout_rate2=0.5,
           pool_size=(2, 2), nb_dense=128):
    diff_input = Input(shape=input_shape)
    rawf_input = Input(shape=input_shape)

    d1 = TSM_Cov2D(diff_input, n_frame, nb_filters1, kernel_size, padding='same', activation='tanh')
    d2 = TSM_Cov2D(d1, n_frame, nb_filters1, kernel_size, padding='valid', activation='tanh')

    r1 = Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv2D(nb_filters1, kernel_size, activation='tanh')(r1)

    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    gated1 = multiply([d2, g1])

    d3 = AveragePooling2D(pool_size)(gated1)
    d4 = Dropout(dropout_rate1)(d3)

    r3 = AveragePooling2D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)

    d5 = TSM_Cov2D(d4, n_frame, nb_filters2, kernel_size, padding='same', activation='tanh')
    d6 = TSM_Cov2D(d5, n_frame, nb_filters2, kernel_size, padding='valid', activation='tanh')

    r5 = Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv2D(nb_filters2, kernel_size, activation='tanh')(r5)

    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    gated2 = multiply([d6, g2])

    d7 = AveragePooling2D(pool_size)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    d9 = Flatten()(d8)
    d10 = Dense(nb_dense, activation='tanh')(d9)
    d11 = Dropout(dropout_rate2)(d10)
    out = Dense(1)(d11)
    model = Model(inputs=[diff_input, rawf_input], outputs=out)
    return model




#%% TSM Multi-tasking

def MTTS_CAN(n_frame, nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3), dropout_rate1=0.25,
                           dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128):

    diff_input = Input(shape=input_shape)
    rawf_input = Input(shape=input_shape)

    d1 = TSM_Cov2D(diff_input, n_frame, nb_filters1, kernel_size, padding='same', activation='tanh')
    d2 = TSM_Cov2D(d1, n_frame, nb_filters1, kernel_size, padding='valid', activation='tanh')

    r1 = Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv2D(nb_filters1, kernel_size, activation='tanh')(r1)

    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    gated1 = multiply([d2, g1])

    d3 = AveragePooling2D(pool_size)(gated1)
    d4 = Dropout(dropout_rate1)(d3)

    r3 = AveragePooling2D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)

    d5 = TSM_Cov2D(d4, n_frame, nb_filters2, kernel_size, padding='same', activation='tanh')
    d6 = TSM_Cov2D(d5, n_frame, nb_filters2, kernel_size, padding='valid', activation='tanh')

    r5 = Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv2D(nb_filters2, kernel_size, activation='tanh')(r5)

    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    gated2 = multiply([d6, g2])

    d7 = AveragePooling2D(pool_size)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    d9 = Flatten()(d8)

    d10_y = Dense(nb_dense, activation='tanh')(d9)
    d11_y = Dropout(dropout_rate2)(d10_y)
    out_y = Dense(1, name='pulse')(d11_y)

    d10_r = Dense(nb_dense, activation='tanh', name='dense_resp')(d9)
    d11_r = Dropout(dropout_rate2, name='dropout_resp')(d10_r)
    out_r = Dense(1, name='resp')(d11_r)

    model = Model(inputs=[diff_input, rawf_input], outputs=[out_y, out_r])
    return model

# model = MTTS_CAN(20, 64, 128, (36, 36, 3))

#%%
def MTTS_CAN_Dual(n_frame, nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3), dropout_rate1=0.25,
                           dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128):

    diff_input = Input(shape=input_shape)
    rawf_input = Input(shape=input_shape)

    d1 = TSM_Cov2D(diff_input, n_frame, nb_filters1, kernel_size, padding='same', activation='tanh')
    d2 = TSM_Cov2D(d1, n_frame, nb_filters1, kernel_size, padding='valid', activation='tanh')

    r1 = Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv2D(nb_filters1, kernel_size, activation='tanh')(r1)

    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    gated1 = multiply([d2, g1])

    g1_rr = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1_rr = Attention_mask()(g1_rr)
    gated1_rr = multiply([d2, g1_rr])


    d3 = AveragePooling2D(pool_size)(gated1)
    d4 = Dropout(dropout_rate1)(d3)

    d3_rr = AveragePooling2D(pool_size)(gated1_rr)
    d4_rr = Dropout(dropout_rate1)(d3_rr)

    r3 = AveragePooling2D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)

    d5 = TSM_Cov2D(d4, n_frame, nb_filters2, kernel_size, padding='same', activation='tanh')
    d6 = TSM_Cov2D(d5, n_frame, nb_filters2, kernel_size, padding='valid', activation='tanh')

    d5_rr = TSM_Cov2D(d4_rr, n_frame, nb_filters2, kernel_size, padding='same', activation='tanh')
    d6_rr = TSM_Cov2D(d5_rr, n_frame, nb_filters2, kernel_size, padding='valid', activation='tanh')


    r5 = Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv2D(nb_filters2, kernel_size, activation='tanh')(r5)

    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    gated2 = multiply([d6, g2])

    g2_rr = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2_rr = Attention_mask()(g2_rr)
    gated2_rr = multiply([d6_rr, g2_rr])

    d7 = AveragePooling2D(pool_size)(gated2)
    d8 = Dropout(dropout_rate1)(d7)
    d9 = Flatten()(d8)

    d7_rr = AveragePooling2D(pool_size)(gated2_rr)
    d8_rr = Dropout(dropout_rate1)(d7_rr)
    d9_rr = Flatten()(d8_rr)

    d10_y = Dense(nb_dense, activation='tanh')(d9)
    d11_y = Dropout(dropout_rate2)(d10_y)
    out_y = Dense(1, name='pulse')(d11_y)

    d10_r = Dense(nb_dense, activation='tanh')(d9_rr)
    d11_r = Dropout(dropout_rate2)(d10_r)
    out_r = Dense(1, name='resp')(d11_r)

    model = Model(inputs=[diff_input, rawf_input], outputs=[out_y, out_r])
    return model

 #%% 3D-CAN

def DeepPhy_3DCNN(n_frame, nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3, 3), dropout_rate1=0.25, dropout_rate2=0.5,
            pool_size=(2, 2, 2), nb_dense=128, use_dataloader=False):

    diff_input = Input(shape=input_shape)
    rawf_input = Input(shape=input_shape)

    d1 = Conv3D(nb_filters1, kernel_size, padding='same', activation='tanh')(diff_input)
    d2 = Conv3D(nb_filters1, kernel_size, activation='tanh')(d1)

    # Appearance Branch
    r1 = Conv3D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv3D(nb_filters1, kernel_size, activation='tanh')(r1)
    g1 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    gated1 = multiply([d2, g1])
    d3 = AveragePooling3D(pool_size)(gated1)

    d4 = Dropout(dropout_rate1)(d3)
    d5 = Conv3D(nb_filters2, kernel_size, padding='same', activation='tanh')(d4)
    d6 = Conv3D(nb_filters2, kernel_size, activation='tanh')(d5)

    r3 = AveragePooling3D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)
    r5 = Conv3D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv3D(nb_filters2, kernel_size, activation='tanh')(r5)
    g2 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    gated2 = multiply([d6, g2])
    d7 = AveragePooling3D(pool_size)(gated2)
    d8 = Dropout(dropout_rate1)(d7)
    d9 = Flatten()(d8)
    d10 = Dense(nb_dense, activation='tanh')(d9)
    d11 = Dropout(dropout_rate2)(d10)
    out = Dense(n_frame)(d11)
    model = Model(inputs=[diff_input, rawf_input], outputs=out)
    return model

input_shape = (36, 36, 10, 3)
model = DeepPhy_3DCNN(10, 32, 64, input_shape)
print('==========================')


#%% MT-3DCAN

def DeepPhy_3DCNN_MT(n_frame, nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3, 3), dropout_rate1=0.25, dropout_rate2=0.5,
            pool_size=(2, 2, 2), nb_dense=128, use_dataloader=False):

    diff_input = Input(shape=input_shape)
    rawf_input = Input(shape=input_shape)

    d1 = Conv3D(nb_filters1, kernel_size, padding='same', activation='tanh')(diff_input)
    d2 = Conv3D(nb_filters1, kernel_size, activation='tanh')(d1)

    # Appearance Branch
    r1 = Conv3D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv3D(nb_filters1, kernel_size, activation='tanh')(r1)
    g1 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    gated1 = multiply([d2, g1])

    d3 = AveragePooling3D(pool_size)(gated1)
    d4 = Dropout(dropout_rate1)(d3)
    d5 = Conv3D(nb_filters2, kernel_size, padding='same', activation='tanh')(d4)
    d6 = Conv3D(nb_filters2, kernel_size, activation='tanh')(d5)

    r3 = AveragePooling3D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)
    r5 = Conv3D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv3D(nb_filters2, kernel_size, activation='tanh')(r5)
    g2 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    gated2 = multiply([d6, g2])
    d7 = AveragePooling3D(pool_size)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    d9 = Flatten()(d8)
    d10_y = Dense(nb_dense, activation='tanh')(d9)
    d11_y = Dropout(dropout_rate2)(d10_y)
    out_y = Dense(n_frame, name='pulse')(d11_y)

    d10_r = Dense(nb_dense, activation='tanh')(d9)
    d11_r = Dropout(dropout_rate2)(d10_r)
    out_r = Dense(n_frame, name='resp')(d11_r)

    model = Model(inputs=[diff_input, rawf_input], outputs=[out_y, out_r])

    return model


#%% Hybrid-CAN

def Hybrid_CAN(n_frame, nb_filters1, nb_filters2, input_shape_1, input_shape_2, kernel_size_1=(3, 3, 3), kernel_size_2=(3,3),
                dropout_rate1=0.25, dropout_rate2=0.5, pool_size_1=(2, 2, 2), pool_size_2=(2, 2),
                nb_dense=128, use_dataloader=False):

    diff_input = Input(shape=input_shape_1)
    rawf_input = Input(shape=input_shape_2)

    # Motion branch
    d1 = Conv3D(nb_filters1, kernel_size_1, padding='same', activation='tanh')(diff_input)
    d2 = Conv3D(nb_filters1, kernel_size_1, activation='tanh')(d1)

    # App branch
    r1 = Conv2D(nb_filters1, kernel_size_2, padding='same', activation='tanh')(rawf_input)
    r2 = Conv2D(nb_filters1, kernel_size_2, activation='tanh')(r1)

    # Mask from App (g1) * Motion Branch (d2)
    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    g1 = K.expand_dims(g1, axis=-1)
    gated1 = multiply([d2, g1])

    # Motion Branch
    d3 = AveragePooling3D(pool_size_1)(gated1)
    d4 = Dropout(dropout_rate1)(d3)
    d5 = Conv3D(nb_filters2, kernel_size_1, padding='same', activation='tanh')(d4)
    d6 = Conv3D(nb_filters2, kernel_size_1, activation='tanh')(d5)

    # App branch
    r3 = AveragePooling2D(pool_size_2)(r2)
    r4 = Dropout(dropout_rate1)(r3)
    r5 = Conv2D(nb_filters2, kernel_size_2, padding='same', activation='tanh')(r4)
    r6 = Conv2D(nb_filters2, kernel_size_2, activation='tanh')(r5)

    # Mask from App (g2) * Motion Branch (d6)
    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    g2 = K.repeat_elements(g2, d6.shape[3], axis=-1)
    g2 = K.expand_dims(g2, axis=-1)
    gated2 = multiply([d6, g2])

    # Motion Branch
    d7 = AveragePooling3D(pool_size_1)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    # Motion Branch
    d9 = Flatten()(d8)
    d10 = Dense(nb_dense, activation='tanh')(d9)
    d11 = Dropout(dropout_rate2)(d10)
    out = Dense(n_frame)(d11)

    model = Model(inputs=[diff_input, rawf_input], outputs=out)
    return model

 #%% MT-Hybrid-CAN

def Hybrid_CAN_MT(n_frame, nb_filters1, nb_filters2, input_shape_1, input_shape_2, kernel_size_1=(3, 3, 3),
                kernel_size_2=(3, 3), dropout_rate1=0.25, dropout_rate2=0.5, pool_size_1=(2, 2, 2), pool_size_2=(2, 2),
                nb_dense=128, use_dataloader=False):

    diff_input = Input(shape=input_shape_1)
    rawf_input = Input(shape=input_shape_2)

    # Motion branch
    d1 = Conv3D(nb_filters1, kernel_size_1, padding='same', activation='tanh')(diff_input)
    d2 = Conv3D(nb_filters1, kernel_size_1, activation='tanh')(d1)

    # App branch
    r1 = Conv2D(nb_filters1, kernel_size_2, padding='same', activation='tanh')(rawf_input)
    r2 = Conv2D(nb_filters1, kernel_size_2, activation='tanh')(r1)

    # Mask from App (g1) * Motion Branch (d2)
    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    # g1 = K.repeat_elements(g1, d2.shape[3], axis=-1)
    g1 = K.expand_dims(g1, axis=-1)
    gated1 = multiply([d2, g1])

    # Motion Branch
    d3 = AveragePooling3D(pool_size_1)(gated1)
    d4 = Dropout(dropout_rate1)(d3)
    d5 = Conv3D(nb_filters2, kernel_size_1, padding='same', activation='tanh')(d4)
    d6 = Conv3D(nb_filters2, kernel_size_1, activation='tanh')(d5)

    # App branch
    r3 = AveragePooling2D(pool_size_2)(r2)
    r4 = Dropout(dropout_rate1)(r3)
    r5 = Conv2D(nb_filters2, kernel_size_2, padding='same', activation='tanh')(r4)
    r6 = Conv2D(nb_filters2, kernel_size_2, activation='tanh')(r5)

    # Mask from App (g2) * Motion Branch (d6)
    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    g2 = K.repeat_elements(g2, d6.shape[3], axis=-1)
    g2 = K.expand_dims(g2, axis=-1)
    gated2 = multiply([d6, g2])

    # Motion Branch
    d7 = AveragePooling3D(pool_size_1)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    # Motion Branch
    d9 = Flatten()(d8)

    d10_y = Dense(nb_dense, activation='tanh')(d9)
    d11_y = Dropout(dropout_rate2)(d10_y)
    out_y = Dense(n_frame, name='pulse')(d11_y)

    d10_r = Dense(nb_dense, activation='tanh')(d9)
    d11_r = Dropout(dropout_rate2)(d10_r)
    out_r = Dense(n_frame, name='resp')(d11_r)

    model = Model(inputs=[diff_input, rawf_input], outputs=[out_y, out_r])
    return model

def Hybrid_CAN_MT_Dual(n_frame, nb_filters1, nb_filters2, input_shape_1, input_shape_2, kernel_size_1=(3, 3, 3),
                kernel_size_2=(3, 3), dropout_rate1=0.25, dropout_rate2=0.5, pool_size_1=(2, 2, 2), pool_size_2=(2, 2),
                nb_dense=128, use_dataloader=False):

    diff_input = Input(shape=input_shape_1)
    rawf_input = Input(shape=input_shape_2)

    # Motion branch
    d1 = Conv3D(nb_filters1, kernel_size_1, padding='same', activation='tanh')(diff_input)
    d2 = Conv3D(nb_filters1, kernel_size_1, activation='tanh')(d1)

    # App branch
    r1 = Conv2D(nb_filters1, kernel_size_2, padding='same', activation='tanh')(rawf_input)
    r2 = Conv2D(nb_filters1, kernel_size_2, activation='tanh')(r1)

    # Mask from App (g1) * Motion Branch (d2)
    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    # g1 = K.repeat_elements(g1, d2.shape[3], axis=-1)
    g1 = K.expand_dims(g1, axis=-1)
    gated1 = multiply([d2, g1])

    # Motion Branch
    d3 = AveragePooling3D(pool_size_1)(gated1)
    d4 = Dropout(dropout_rate1)(d3)
    d5 = Conv3D(nb_filters2, kernel_size_1, padding='same', activation='tanh')(d4)
    d6 = Conv3D(nb_filters2, kernel_size_1, activation='tanh')(d5)

    # App branch
    r3 = AveragePooling2D(pool_size_2)(r2)
    r4 = Dropout(dropout_rate1)(r3)
    r5 = Conv2D(nb_filters2, kernel_size_2, padding='same', activation='tanh')(r4)
    r6 = Conv2D(nb_filters2, kernel_size_2, activation='tanh')(r5)

    # Mask from App (g2) * Motion Branch (d6)
    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    g2 = K.repeat_elements(g2, d6.shape[3], axis=-1)
    g2 = K.expand_dims(g2, axis=-1)
    gated2 = multiply([d6, g2])


    g2_rr = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2_rr = Attention_mask()(g2_rr)
    g2_rr = K.repeat_elements(g2_rr, d6.shape[3], axis=-1)
    g2_rr = K.expand_dims(g2_rr, axis=-1)
    gated2_rr = multiply([d6, g2_rr])

    # Motion Branch
    d7 = AveragePooling3D(pool_size_1)(gated2)
    d8 = Dropout(dropout_rate1)(d7)
    d7_rr = AveragePooling3D(pool_size_1)(gated2_rr)
    d8_rr = Dropout(dropout_rate1)(d7_rr)

    # Motion Branch
    d9 = Flatten()(d8)
    d9_rr = Flatten()(d8_rr)

    d10_y = Dense(nb_dense, activation='tanh')(d9)
    d11_y = Dropout(dropout_rate2)(d10_y)
    out_y = Dense(n_frame, name='pulse')(d11_y)

    d10_r = Dense(nb_dense, activation='tanh')(d9_rr)
    d11_r = Dropout(dropout_rate2)(d10_r)
    out_r = Dense(n_frame, name='resp')(d11_r)

    model = Model(inputs=[diff_input, rawf_input], outputs=[out_y, out_r])
    return model

# input_shape_1 = (36, 36, 10, 3) # Motion
# input_shape_2 = (36, 36, 3) # Apperance
#
# model = DeepPhysMix(32, 64, input_shape_1, input_shape_2)



#%%
def DeepPhysMotion(n_frame, nb_filters1, nb_filters2, input_shape_1, kernel_size_1=(3, 3, 3),
                dropout_rate1=0.25, dropout_rate2=0.5, pool_size_1=(2, 2, 2), nb_dense=128):

    diff_input = Input(shape=input_shape_1)

    # Motion branch
    d1 = Conv3D(nb_filters1, kernel_size_1, padding='same', activation='tanh')(diff_input)
    d2 = Conv3D(nb_filters1, kernel_size_1, activation='tanh')(d1)

    # Motion Branch
    d3 = AveragePooling3D(pool_size_1)(d2)
    d4 = Dropout(dropout_rate1)(d3)
    d5 = Conv3D(nb_filters2, kernel_size_1, padding='same', activation='tanh')(d4)
    d6 = Conv3D(nb_filters2, kernel_size_1, activation='tanh')(d5)

    # Motion Branch
    d7 = AveragePooling3D(pool_size_1)(d6)
    d8 = Dropout(dropout_rate1)(d7)

    # Motion Branch
    d9 = Flatten()(d8)
    d10 = Dense(nb_dense, activation='tanh')(d9)
    d11 = Dropout(dropout_rate2)(d10)
    out = Dense(n_frame)(d11)

    model = Model(inputs=[diff_input], outputs=out)
    return model



#%% TSM without Attention and App branch

def DeepPhys_TSM_Motion(n_frame, nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3), dropout_rate1=0.25, dropout_rate2=0.5,
            pool_size=(2, 2), nb_dense=128):

    diff_input = Input(shape=input_shape)
    rawf_input = Input(shape=input_shape)

    d1 = TSM_Cov2D(diff_input, n_frame, nb_filters1, kernel_size, padding='same', activation='tanh')
    d2 = TSM_Cov2D(d1, n_frame, nb_filters1, kernel_size, padding='valid', activation='tanh')

    d3 = AveragePooling2D(pool_size)(d2)
    d4 = Dropout(dropout_rate1)(d3)

    d5 = TSM_Cov2D(d4, n_frame, nb_filters2, kernel_size, padding='same', activation='tanh')
    d6 = TSM_Cov2D(d5, n_frame, nb_filters2, kernel_size, padding='valid', activation='tanh')

    d7 = AveragePooling2D(pool_size)(d6)
    d8 = Dropout(dropout_rate1)(d7)

    d9 = Flatten()(d8)
    d10 = Dense(nb_dense, activation='tanh')(d9)
    d11 = Dropout(dropout_rate2)(d10)
    out = Dense(1)(d11)
    model = Model(inputs=[diff_input, rawf_input], outputs=out)
    return model
#%%
class HeartBeat(keras.callbacks.Callback):
    def __init__(self, train_gen, test_gen, args, cv_split, save_dir):
        super(HeartBeat, self).__init__()
        self.train_gen = train_gen
        self.test_gen = test_gen
        self.args = args
        self.cv_split = cv_split
        self.save_dir = save_dir

    def on_epoch_end(self, epoch, logs={}):
        # if epoch > 45:
        #     print(' | Predicting HeartBeat')
        #     yptrain = self.model.predict(self.train_gen, verbose=1)
        #     scipy.io.savemat(self.save_dir + '/yptrain' + str(epoch) + '_cv' + self.cv_split +'.mat',
        #                      mdict={'yptrain': yptrain})
        #     yptest = self.model.predict(self.test_gen, verbose=1)
        #     scipy.io.savemat(self.save_dir + '/yptest' + str(epoch) + '_cv' + self.cv_split + '.mat',
        #                      mdict={'yptest': yptest})
        print('PROGRESS: 0.00%')
