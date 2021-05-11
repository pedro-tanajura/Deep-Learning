import cv2
import os, os.path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
 
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
from keras.models import Sequential, Model,load_model
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from google.colab.patches import cv2_imshow
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D, Dropout
from keras.preprocessing import image
from keras.initializers import glorot_uniform
 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.applications.resnet import ResNet50
from keras.applications import DenseNet121, VGG16, VGG19, Xception
 
train_path="/content/drive/MyDrive/X-Ray Dataset/Train/Grupo 1"
test_path="/content/drive/MyDrive/X-Ray Dataset/Test/Grupo 1"
class_names=os.listdir(train_path)
class_names_test=os.listdir(test_path)
'''
train_datagen = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode="reflect",
    horizontal_flip=False,
    vertical_flip=False,
)
'''
train_datagen = ImageDataGenerator(rotation_range=0.15,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.15,
                                  horizontal_flip=True,
                                  zoom_range=0.2,
                                  rescale=1./255,
					                        vertical_flip=True,
                                  fill_mode="reflect")
 
#train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator(rescale=1./255)
#Cria lotes de novas imagens feitas por DA
batch = 64
train_generator = train_datagen.flow_from_directory(train_path,target_size=(224, 224),batch_size=batch,shuffle=True,class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_path, target_size=(224,224),batch_size=batch,shuffle=False, class_mode='categorical')
 
print("indice das classes (treino):\n", train_generator.class_indices)
print("indice das classes (teste):\n", test_generator.class_indices)
 
#base_model = Xception(include_top=False, weights='/content/drive/MyDrive/CNN_pesos/Xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape=(224, 224,3),pooling='avg')
#base_model = VGG19(include_top=False, weights='/content/drive/MyDrive/CNN_pesos/VGG19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape=(224, 224,3),pooling='avg')
base_model = VGG16(include_top=False, weights='/content/drive/MyDrive/CNN_pesos/VGG16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape=(224, 224,3),pooling='avg')
#base_model = ResNet50(include_top=False, weights='/content/drive/MyDrive/CNN_pesos/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape=(224, 224,3),pooling='avg')
#base_model = DenseNet121(include_top=False, weights='/content/drive/MyDrive/CNN_pesos/densenet121/DenseNet-BC-121-32-no-top.h5', input_shape=(224, 224,3),pooling='avg')
 
METRICS = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]
 
opt = Adam(lr=1e-4, decay=1e-4 / 10)
#opt = SGD(lr=1e-4, decay=1e-4 / 10)
 
#Final da CNN
headModel = base_model.output
headModel = Flatten()(headModel)
headModel = Dense(256, activation='relu', name='fc1',kernel_initializer=glorot_uniform(seed=0))(headModel)
headModel = Dense(128, activation='relu', name='fc2',kernel_initializer=glorot_uniform(seed=0))(headModel)
headModel = Dense(4,activation='sigmoid', name='fc3',kernel_initializer=glorot_uniform(seed=0))(headModel)
 
model = Model(inputs=base_model.input, outputs=headModel)
 
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=METRICS)
 
es=EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=15)
 
#mc = ModelCheckpoint('/content/drive/MyDrive/CNN_pesos/Xception/best_model.h5', monitor='val_accuracy', save_best_only=True)
#mc = ModelCheckpoint('/content/drive/MyDrive/CNN_pesos/VGG19/best_model.h5', monitor='val_accuracy', save_best_only=True)
mc = ModelCheckpoint('/content/drive/MyDrive/CNN_pesos/VGG16/best_model_DA123.h5', monitor='val_accuracy', save_best_only=True)
#mc = ModelCheckpoint('/content/drive/MyDrive/CNN_pesos/resnet50/best_model.h5', monitor='val_accuracy', save_best_only=True)
#mc = ModelCheckpoint('/content/drive/MyDrive/CNN_pesos/densenet121/best_model.h5', monitor='val_accuracy', save_best_only=True)
 
epocas = 100
 
H = model.fit(train_generator, validation_data=test_generator, epochs=epocas, verbose=1, callbacks=[mc,es], batch_size=batch
            ,steps_per_epoch=len(train_generator.filenames) // batch)
 
#model.load_weights('/content/drive/MyDrive/CNN_pesos/VGG19/best_model.h5')
#model.load_weights("/content/drive/MyDrive/CNN_pesos/resnet50/best_model.h5")
model.load_weights('/content/drive/MyDrive/CNN_pesos/VGG16/best_model_DA123.h5') #DA foi melhor
#model.load_weights("/content/drive/MyDrive/CNN_pesos/densenet121/best_model.h5")
#model.load_weights('/content/drive/MyDrive/CNN_pesos/Xception/best_model.h5')  #DA foi melhor
 
model.evaluate(test_generator)
 
# get the ground truth of your data. 
test_labels=test_generator.classes 
# predict the probability distribution of the data
predictions=model.predict(test_generator, batch_size = batch)
# get the class with highest probability for each sample
y_pred = np.argmax(predictions, axis=-1)
print(classification_report(test_labels, y_pred, target_names=train_generator.class_indices))
 
plt.style.use("ggplot")
plt.figure()
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.plot(H.history["accuracy"], label="train_accuracy")
plt.plot(H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.show()
