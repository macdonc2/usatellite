from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Lambda, RepeatVector, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt

from loss.loss_metrics import f1
import tensorflow as tf

import tensorflow.keras as k


class Unet2d:

    """
    Constructs a unet using 2D convolutional filters.  Args are: 
    n_filters: number of filters at first layer.
    kernel_size: convolutional kernel size.
    l2_lambda: optional l2 kernel regularizer weight.  Only l2 regularization for now.
    bathnorm: whether you want to standardize batches.
    dropout: dropout rate to prevent overfitting.  This is probably more important as number of feature maps increases.
    """

    def __init__(self, input_img, n_filters=32, kernel_size=3, l2_lambda=0, batchnorm=True, dropout=0.5):

        self._input_img = Input((input_img.shape[1], input_img.shape[2],input_img.shape[3]), name='img')
        self._n_filters = n_filters
        self._kernel_size = kernel_size
        self._l2_lambda = l2_lambda
        self._batchnorm = batchnorm
        self._dropout = dropout
        #self._model = self.get_unet()

        #self._model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=["categorical_accuracy", Loss.f1])
        #self._model.summary()

    def conv2d_block(self, input_tensor, n_filters):

        '''
        This allows the user to simplify how blocks of convolutional layers are used.  A typical convolutional layer has 
        3 convolutional blocks.  Who knows?  Who doesn't?  Works well.  Most agruments that users can control are in 
        the class initilization.
        '''

        # first layer
        x = Conv2D(filters=n_filters, kernel_size=(self._kernel_size, self._kernel_size), kernel_initializer="he_normal",
                padding="same", kernel_regularizer=regularizers.l2(self._l2_lambda))(input_tensor)
        if self._batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # second layer
        x = Conv2D(filters=n_filters, kernel_size=(self._kernel_size, self._kernel_size), kernel_initializer="he_normal",
                padding="same", kernel_regularizer=regularizers.l2(self._l2_lambda))(x)
        if self._batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # third layer
        x = Conv2D(filters=n_filters, kernel_size=(self._kernel_size, self._kernel_size), kernel_initializer="he_normal",
                padding="same", kernel_regularizer=regularizers.l2(self._l2_lambda))(x)
        if self._batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x



    def get_unet(self):

        """
        Construct a UNet with a predefined architecture.  In the future, the user should be allowed to specify the
        image input and output sizes as well as number of layers.  
        """

        # contracting path
        c1 = self.conv2d_block(self._input_img, self._n_filters*1)
        p1 = MaxPooling2D((2, 2)) (c1)
        #p1 = Dropout(self._dropout)(p1)

        c2 = self.conv2d_block(p1, self._n_filters*2)
        p2 = MaxPooling2D((2, 2)) (c2)
        #p2 = Dropout(self._dropout)(p2)

        c3 = self.conv2d_block(p2, self._n_filters*4)
        p3 = MaxPooling2D((2, 2)) (c3)
        #p3 = Dropout(self._dropout)(p3)

        c4 = self.conv2d_block(p3, self._n_filters*8)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
        #p4 = Dropout(self._dropout)(p4)

        c5 = self.conv2d_block(p4, self._n_filters*16)
        c5 = Dropout(self._dropout)(c5)

        # expansive path
        u6 = Conv2DTranspose(self._n_filters, (3, 3), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        #u6 = Dropout(self._dropout)(u6)
        c6 = self.conv2d_block(u6, self._n_filters*8)

        u7 = Conv2DTranspose(self._n_filters, (3, 3), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        #u7 = Dropout(self._dropout)(u7)
        c7 = self.conv2d_block(u7, self._n_filters*4)

        u8 = Conv2DTranspose(self._n_filters, (3, 3), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        #u8 = Dropout(self._dropout)(u8)
        c8 = self.conv2d_block(u8, self._n_filters*2)

        u9 = Conv2DTranspose(self._n_filters, (3, 3), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        #u9 = Dropout(self._dropout)(u9)
        c9 = self.conv2d_block(u9, self._n_filters*1)

        outputs = Conv2D(6, (1, 1), activation='softmax') (c9)
        self._model = Model(inputs=[self._input_img], outputs=[outputs])

        return self._model

    def get_deeper_unet(self):

        """
        Construct a UNet with a predefined architecture.  In the future, the user should be allowed to specify the
        image input and output sizes as well as number of layers.  
        """

        # contracting path
        c1 = self.conv2d_block(self._input_img, self._n_filters*1)
        p1 = MaxPooling2D((2, 2)) (c1)
        #p1 = Dropout(self._dropout)(p1)

        c2 = self.conv2d_block(p1, self._n_filters*2)
        p2 = MaxPooling2D((2, 2)) (c2)
        #p2 = Dropout(self._dropout)(p2)

        c3 = self.conv2d_block(p2, self._n_filters*4)
        p3 = MaxPooling2D((2, 2)) (c3)
        #p3 = Dropout(self._dropout)(p3)

        c4 = self.conv2d_block(p3, self._n_filters*8)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
        #p4 = Dropout(self._dropout)(p4)

        c5 = self.conv2d_block(p4, self._n_filters*16)
        p5 = MaxPooling2D(pool_size=(2, 2)) (c5)
        #p5 = Dropout(self._dropout)(p4)

        c6 = self.conv2d_block(p5, self._n_filters*32)
        p6 = MaxPooling2D(pool_size=(2, 2)) (c6)
        #p5 = Dropout(self._dropout)(p4)

        c7 = self.conv2d_block(p6, self._n_filters*64)
        p7 = MaxPooling2D(pool_size=(2, 2)) (c7)
        #p5 = Dropout(self._dropout)(p4)

        c8 = self.conv2d_block(p7, self._n_filters*128)
        c8 = Dropout(self._dropout)(c8)

        # expansive path
        u9 = Conv2DTranspose(self._n_filters, (3, 3), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c7])
        #u6 = Dropout(self._dropout)(u6)
        c9 = self.conv2d_block(u9, self._n_filters*64)

        # expansive path
        u10 = Conv2DTranspose(self._n_filters, (3, 3), strides=(2, 2), padding='same') (c9)
        u10 = concatenate([u10, c6])
        #u6 = Dropout(self._dropout)(u6)
        c10 = self.conv2d_block(u10, self._n_filters*32)

        # expansive path
        u11 = Conv2DTranspose(self._n_filters, (3, 3), strides=(2, 2), padding='same') (c10)
        u11 = concatenate([u11, c5])
        #u6 = Dropout(self._dropout)(u6)
        c11 = self.conv2d_block(u11, self._n_filters*16)

        # expansive path
        u12 = Conv2DTranspose(self._n_filters, (3, 3), strides=(2, 2), padding='same') (c11)
        u12 = concatenate([u12, c4])
        #u6 = Dropout(self._dropout)(u6)
        c12 = self.conv2d_block(u12, self._n_filters*8)

        u13 = Conv2DTranspose(self._n_filters, (3, 3), strides=(2, 2), padding='same') (c12)
        u13 = concatenate([u13, c3])
        #u7 = Dropout(self._dropout)(u7)
        c13 = self.conv2d_block(u13, self._n_filters*4)

        u14 = Conv2DTranspose(self._n_filters, (3, 3), strides=(2, 2), padding='same') (c13)
        u14 = concatenate([u14, c2])
        #u8 = Dropout(self._dropout)(u8)
        c14 = self.conv2d_block(u14, self._n_filters*2)

        u15 = Conv2DTranspose(self._n_filters, (3, 3), strides=(2, 2), padding='same') (c14)
        u15 = concatenate([u15, c1], axis=3)
        #u9 = Dropout(self._dropout)(u9)
        c15 = self.conv2d_block(u15, self._n_filters*1)

        outputs = Conv2D(6, (1, 1), activation='softmax') (c15)
        self._model = Model(inputs=[self._input_img], outputs=[outputs])

        return self._model


    def callback_list(self,patience=10, verbose=1, factor=1, min_lr=0.00001, save_best_only=True,
            save_weights_only=True):

        callbacks = [
        EarlyStopping(monitor=f1, mode='max', patience=20, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
        ModelCheckpoint('model-test-focal.h5', verbose=1, save_best_only=True, save_weights_only=True)
        ]

        return callbacks

class PlotLosses(k.callbacks.Callback):

    """
    Helper loss plot from: https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e

    Thanks, Piotr!

    This class will simply help output some loss plots.  Need to sort out how to merge this with 
    callback_list, so the user doesn't have to think about it.
    """


    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()