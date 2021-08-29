from keras.models import Model
from keras import regularizers
from keras.models import Sequential
from keras.layers import merge, Input, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, \
    Concatenate

from VGGish_Model.vggish import VGGish

def vgg16():
    # 用于正则化时权重降低的速度
    weight_decay = 0.0005  # 权重衰减（L2正则化），作用是避免过拟合
    # layer1 32*32*3
    model = Sequential()
    # 第一层 卷积层 的卷积核的数目是32 ，卷积核的大小是3*3，keras卷积层stride默认是1*1
    # 对于stride=1*1,padding ='same',这种情况卷积后的图像shape与卷积前相同，本层后shape还是32*32
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1),
                     input_shape=(224, 224, 3), kernel_regularizer=regularizers.l2(weight_decay)))

    model.add(Activation('relu'))
    # 进行一次归一化
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    #layer2 32*32*64
    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #keras 池化层 stride默认是2*2, padding默认是valid，输出的shape是16*16*64
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same'))

    #layer3 16*16*64
    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    #layer4 16*16*128
    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #layer5 8*8*256
    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    #layer6 8*8*256
    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    #layer7 8*8*256
    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #layer8 4*4*512
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    #layer9 4*4*512
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    #layer10 4*4*512
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #layer11 2*2*512
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    #layer12 2*2*512
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    #layer13 2*2*512
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    #layer14 1*1*512 全连接
    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    #layer15 512
    model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    #layer16 512
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


def generate_vgg16():
    input_shape = (224, 224, 3)
    model = Sequential([
        Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
		MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='rel u'),
        Dense(4096, activation='relu'),
        Dense(1000, activation='softmax')
    ])

    return model

def merge_cnn():
    CNN_left  = vgg16()
    CNN_right = vgg16()

    input_left  = CNN_left.input
    input_right = CNN_right.input

    merged = Concatenate(axis=1)([CNN_left.output, CNN_right.output])
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    fc = Dense(512, activation='relu')(merged)
    fc = Dropout(0.5)(fc)
    out = Dense(2, activation='softmax')(fc)

    # Return the model object
    model = Model(inputs=[input_left, input_right], outputs=out)
    return model

def merge_camera_voice_command():
    CNN_left = vgg16()
    CNN_right = vgg16()
    CNN_audio = VGGish()

    input_left = CNN_left.input
    input_right = CNN_right.input
    input_audio = CNN_audio.input

    print(input_left.shape, input_audio.shape)

    merged = Concatenate(axis=1)([CNN_left.output, CNN_right.output, CNN_audio.output])
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    fc = Dense(512, activation='relu')(merged)
    fc = Dropout(0.5)(fc)
    out = Dense(2, activation='softmax')(fc)

    # Return the model object
    model = Model(inputs=[input_left, input_right, input_audio], outputs=out)
    #model.summary()
    return model

if __name__ == '__main__':
    # merge_cnn()
    merge_camera_voice_command()
    # model.summary()