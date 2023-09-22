import metrics_self
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing  import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import LSTM, Dense,Activation, Bidirectional,Dropout
from keras.losses import mean_squared_error
from keras.models import Sequential
static_no_shuffle=input_structure_paramorigin=pd.read_csv('./train_static.csv')
static=shuffle(static_no_shuffle,random_state=1337) #random_state有了这玩意，数据就不会变了
sc = MinMaxScaler(feature_range=(0, 1))
input_static=np.array(static.iloc[:,0:4])
output_static=np.array(static.iloc[:,4:7])
x_train,x_test,y_train,y_test=train_test_split(input_static,output_static,test_size=0.25,random_state=42)
input_static[1],output_static[1]
sc = MinMaxScaler(feature_range=(0, 1))
x_train_scale = sc.fit_transform(x_train)
x_test_scale = sc.transform(x_test)

sc_y = MinMaxScaler(feature_range=(0, 1))
y_train_scale = sc_y.fit_transform(y_train)
y_test_scale = sc_y.transform(y_test)


class skip_con(tf.keras.layers.Layer):
    def __init__(self, in_fan, out_fan):
        super(skip_con,self).__init__()
        self.in_fan=in_fan
        self.out_fan=out_fan
        self.liner=tf.keras.layers.Dense(out_fan,use_bias=False,activation='relu')
        self.transform=tf.keras.layers.Dense(in_fan,use_bias=False)
        self.bn1=tf.keras.layers.BatchNormalization()
        self.bn2=tf.keras.layers.BatchNormalization()
    def call(self,x):
        _x=self.liner(x)
        if self.in_fan==self.out_fan:
            return self.bn1(x+_x)
        elif self.in_fan!=self.out_fan:
            x_=self.transform(_x)
            return self.bn2(x+x_)
    def get_config(self):
        config = super(skip_con, self).get_config()
        config.update({"in_fan":self.in_fan,
                       "out_fan":self.out_fan})
        return config
        
model=Sequential()
model.add(Dense(128,input_shape=(x_train_scale.shape[1],)
                ,activation='relu'))
model.add(Skip_con(128,64))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(3,activation='relu'))
model.compile(loss='mae',optimizer='rmsprop')
batchsize = 32
history = model.fit(x_train_scale, y_train_scale, epochs=500, batch_size=batchsize, validation_data=(x_test_scale, y_test_scale))

model.save('weinian.h5')
model.save_weights('weinian_checkpoint')
model.summary()

model.save('weinian.h5')
model.save_weights('weinian_checkpoint')