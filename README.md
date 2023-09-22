# optimize_honeycomb\model_save proble

model_save:
当class为自定义，然后需要保存为.h5时，需要将class添加一个def get_config(self)的子类部分，进行更行class的参数然后以字典的形式保存：
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
