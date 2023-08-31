#!/usr/bin/python
import sys
sys.path.append('../src')


root_dir='../boveda';
col_id_x=0;
col_id_y=1;


import tensorflow as tf
import pandas as pd

df=pd.read_csv(root_dir+'/dataset.csv');

from DataGeneratorTool.MultiSpectralFromDataframe import DataGeneratorFromDataframe

dat_gen=DataGeneratorFromDataframe( df,col_id_x,col_id_y,
                                    root_dir=root_dir,
                                    rotation_range=0,
                                    vertical_flip=False,
                                    horizontal_flip=False,
                                    zoom_range=None,
                                    batch_size=2,
                                    shuffle=True);

print(df)
print(issubclass(DataGeneratorFromDataframe,tf.keras.utils.Sequence))
              
