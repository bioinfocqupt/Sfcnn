#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, Dense, Flatten, Activation, BatchNormalization
import tensorflow as tf
import pickle
import numpy as np
import os
import sys
import h5py
import tensorflow.keras.backend as K


# In[ ]:
#Load the data of training set and test set. The last ~10% of the training set is split as the validation set
def load_file():
    with open('train_grids.pkl','rb') as f:
        train_grids = pickle.load(f)
    with open('train_label.pkl','rb') as f:
        train_label = pickle.load(f)
    assert train_grids.shape[0] == len(train_label)
    val_x = train_grids[41000:]
    val_y = train_label[41000:]
    mask1 = list(range(41000))
    np.random.seed(1234)
    np.random.shuffle(mask1)
    train_x = train_grids[mask1]
    train_y = train_label[mask1]

    mask2 = list(range(len(val_y)))
    np.random.shuffle(mask2)
    val_x = val_x[mask2]
    val_y = val_y[mask2]
    with open('core_grids.pkl','rb') as f:
        core_grids = pickle.load(f)
    with open('core_label.pkl','rb') as f:
        core_label = pickle.load(f)
    test_x = core_grids
    test_y = core_label

    return (train_x, train_y, val_x, val_y, test_x, test_y)

train_x, train_y, val_x, val_y, test_x, test_y = load_file()

#Normalize the label
if True:
    train_y = train_y / 15.0
    test_y = test_y / 15.0
    val_y = val_y / 15.0

# In[ ]:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch', '-b', default=64, type=int)
parser.add_argument('--dropout', '-d', default=0.5, type=float)
parser.add_argument('--lr', default=0.004, type=float)
args = parser.parse_args()

batch_size = args.batch
epoch = 200
#Build the 3d cnn model. 
model = tf.keras.Sequential([
    Conv3D(7,kernel_size=(1,1,1),input_shape=(20,20,20,28),strides=(1,1,1)),
    BatchNormalization(),  
    Activation(tf.nn.relu),
    Conv3D(7,kernel_size=(3,3,3)),
    BatchNormalization(),  
    Activation(tf.nn.relu),
    Conv3D(7,kernel_size=(3,3,3)),
    BatchNormalization(),
    Activation(tf.nn.relu),
    Conv3D(28,kernel_size=(1,1,1)),
    BatchNormalization(),  
    Activation(tf.nn.relu),
    Conv3D(56,kernel_size=(3,3,3),padding='same'),
    BatchNormalization(),  
    Activation(tf.nn.relu),
    MaxPooling3D(pool_size=2),
    Conv3D(112,kernel_size=(3,3,3),padding='same'),
    BatchNormalization(),  
    Activation(tf.nn.relu),
    MaxPooling3D(pool_size=2),
    Conv3D(224,kernel_size=(3,3,3),padding='same'),
    BatchNormalization(),  
    Activation(tf.nn.relu),
    MaxPooling3D(pool_size=2),
    Flatten(),
    Dense(256),
    BatchNormalization(),
    Activation(tf.nn.relu),
    Dropout(args.dropout),
    Dense(1,kernel_regularizer=tf.keras.regularizers.l2(0.01))]
)

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(args.lr),
    loss='mse',
    metrics=['mae']
)

filepath = 'cnnmodel/weights_{epoch:03d}-{val_loss:.4f}.h5'
if not os.path.exists('cnnmodel'):
    os.mkdir('cnnmodel')

hist = model.fit(
    train_x,
    train_y,
    batch_size=batch_size,
    epochs=epoch,
    callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')],
    validation_data=(val_x,val_y)
)

loss, mae = model.evaluate(
    test_x,
    test_y,
    batch_size=batch_size
)





