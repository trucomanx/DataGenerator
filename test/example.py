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
                                    rotation_range=15,
                                    vertical_flip=True,
                                    horizontal_flip=True,
                                    zoom_range=[0.8,1.2],
                                    batch_size=2,
                                    shuffle=False);

print(df)
print('issubclass(Sequence)?',issubclass(DataGeneratorFromDataframe,tf.keras.utils.Sequence))


import matplotlib.pyplot as plt
Nlin=5;
Ncol=8;
Nch=0;
BatchId=0;

fig, axs = plt.subplots(Nlin,Ncol);
for l in range(Nlin):
    for c in range(Ncol):
        X,y=dat_gen.__getitem__(BatchId);
        #print(l,c);
        #print(dat_gen.get_last_batch_indexes());
        #print(dat_gen.get_last_transform());
        #axs[l,c].set_title(str(l)+','+str(c))
        axs[l,c].imshow(X[0][:,:,Nch]);
plt.show();

