import os
import glob
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense,Activation, BatchNormalization
import keras
from keras.callbacks import LearningRateScheduler

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 45:
        lrate = 0.0005
    if epoch > 75:
        lrate = 0.0003
    if epoch> 100:
        lrate=0.0001
    return lrate

def show_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()
    
 
def show_images_labels_predictions(images,labels,predictions,start_id,num=10):
    plt.gcf().set_size_inches(12, 14)
    if num>25: num=25 
    idxs = np.random.randint(0, len(test_feature), size=25)
    for i in range(num):
    
        ax=plt.subplot(5,5,i+1)
        ax.imshow(images[idxs[i]], cmap='binary')
        
        # 有 AI 預測結果資料, 才在標題顯示預測結果
        if( len(predictions) > 0 ) :
            title = 'ai = ' + str(predictions[idxs[i]])
            # 預測正確顯示(o), 錯誤顯示(x)
            title += (' (o)' if predictions[idxs[i]]==labels[idxs[i]] else ' (x)') 
            title += '\nlabel = ' + str(labels[idxs[i]])
        # 沒有 AI 預測結果資料, 只在標題顯示真實數值
        else :
            title = 'label = ' + str(labels[idxs[i]])
            
        # X, Y 軸不顯示刻度    
        ax.set_title(title,fontsize=12) 
        ax.set_xticks([]);ax.set_yticks([])        
        start_id+=1 
    plt.show()



(train_feature, train_label), (test_feature, test_label) = cifar10.load_data()
print(train_feature.shape, train_label.shape)


train_feature_normalize = train_feature/255
test_feature_normalize = test_feature/255

train_label_onehot = np_utils.to_categorical(train_label, num_classes=10)
test_label_onehot = np_utils.to_categorical(test_label, num_classes=10)

weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
 
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
 
model.add(Flatten())
model.add(Dense(units=10, activation='softmax'))
"""
model = Sequential()

model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 padding='same',
                 input_shape = (32,32,3),
                 activation='relu'))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# conv block 2
model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu") )
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# dense block
model.add(Flatten())
model.add(Dense(units=512, activation="relu") )
model.add(Dropout(0.5))
model.add(Dense(units=10,activation='softmax'))
"""

model.summary()
 
#data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
    )
datagen.fit(train_feature_normalize)
 
#training
batch_size = 64
 
#opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
#model.compile(loss='categorical_crossentropy',optimizer='adam' , metrics=['accuracy'])

opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy',optimizer=opt_rms , metrics=['accuracy'])

try:
    model.load_weights("Cifar10_CNN_model_3.weight")
    print("載入模型參數成功，繼續訓練模型!")
except :    
    print("載入模型參數失敗，開始訓練一個新模型!")

model.fit_generator(datagen.flow(train_feature_normalize, train_label_onehot, batch_size=batch_size),\
                    steps_per_epoch=train_feature.shape[0] // batch_size,epochs=125,\
                    verbose=1,validation_data=(test_feature_normalize, test_label_onehot),callbacks=[LearningRateScheduler(lr_schedule)])




scores = model.evaluate(test_feature_normalize, test_label_onehot, batch_size=128, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))

prediction = model.predict_classes(test_feature_normalize)
show_images_labels_predictions(test_feature,test_label,prediction,0,20)

model.save('Cifar10_CNN_model_3.h5')     #將模型儲存至 HDF5檔案中
print("\Cifar10_CNN_model_3.h5 模型儲存完畢!")
model.save_weights("Cifar10_CNN_model_3.weight") # 將參數儲存不含模型
print("模型參數儲存完畢!")

del model