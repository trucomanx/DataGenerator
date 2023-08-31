import sys
import tensorflow as tf
import DataGeneratorTool.MultiSpectralTool as DGMST
import DataGeneratorTool.LoadDataframe as DGLDF
import numpy as np

class DataGeneratorFromDataframe(tf.keras.utils.Sequence):
    '''
    Generates data for Keras.
    
    Parameters
    
    df: dataframe :
    col_id_x: int :
    col_id_y: int :
    rotation_range: float : 
    vertical_flip: bool : First dimmension
    horizontal_flip: bool : Second dimmension
    zoom_range: float :
    batch_size: int :
    shuffle: bool :
    
    return
    DataGenerator
    
    '''
    def __init__(   self,
                    df,
                    col_id_x,
                    col_id_y,
                    root_dir,
                    rotation_range=0,
                    vertical_flip=False, #first dimmension
                    horizontal_flip=False, #second dimmension
                    zoom_range=None,
                    batch_size=32,
                    shuffle=True):
        'Initialization'
        self.df = df;
        self.col_id_x = col_id_x;
        self.col_id_y = col_id_y;
        self.rotation_range = rotation_range;
        self.horizontal_flip = horizontal_flip;
        self.vertical_flip = vertical_flip;
        self.zoom_range = zoom_range;
        self.batch_size = int(batch_size);
        self.shuffle = shuffle;
        self.root_dir = root_dir;
        
        self.last_transform={
            "rotation_angle": None,
            "horizontal_flip": None,
            "vertical_flip": None,
            "zoom_factor": None
        }
        self.last_batch_indexes=None;
        
        self.L = df.shape[0];
        if self.batch_size>self.L:
            print(' ')
            print('ERROR in file : '+__file__)
            print('ERROR in class: '+__name__)
            sys.exit('ERROR: parameter batch_size greather than number of elements');

        self.NBatchsPerEpoch=int(np.ceil(self.L / self.batch_size));

        self.on_epoch_end()

    def __len__(self):
        'OBLIGATORIO: Denotes the number of batches per epoch'
        return self.NBatchsPerEpoch;


    def __getitem__(self, index):
        'OBLIGATORIO: Generate one batch of data'
        if index >= (self.NBatchsPerEpoch-1):
            end =self.L;
            init=end-self.batch_size;
            batch_indexes=self.indexes[init:end];
        else:
            init=index*self.batch_size;
            end =init+self.batch_size;
            batch_indexes=self.indexes[init:end];
        
        self.last_transform["horizontal_flip"] = False;
        self.last_transform["vertical_flip"] = False;
        self.last_transform["zoom_factor"] = 1.0;
        self.last_transform["rotation_angle"] = 0.0;
        
        if self.horizontal_flip==True and np.random.binomial(1,0.5)==1:
            self.last_transform["horizontal_flip"] = True;
            
        if self.vertical_flip==True and np.random.binomial(1,0.5)==1:
            self.last_transform["vertical_flip"] = True;
            
        if self.zoom_range!=None:
            zoom_factor=np.random.uniform(self.zoom_range[0],self.zoom_range[1],1)[0];
            self.last_transform["zoom_factor"] = zoom_factor;
        
        if self.rotation_range!=0:
            angle=self.rotation_range*np.random.uniform(-1.0,1.0,1)[0];
            self.last_transform["rotation_angle"] = angle;
            
        
        self.last_batch_indexes=batch_indexes;
        X, y = self.get_item_data_generation(batch_indexes,self.last_transform)

        return X, y

    def on_epoch_end(self):
        'OPCIONAL: Updates indexes after each epoch'
        self.indexes = np.arange(self.L)

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

        return;

    def get_last_transform(self):
        return self.last_transform;
    def get_last_batch_indexes(self):
        return self.last_batch_indexes;
    
    def get_item_data_generation(self,batch_indexes,last_transform):
        'Generates data containing batch_size samples'
        # Initialization
        
        batch_df = self.df.iloc[batch_indexes,:]
        X, y = DGLDF.load_numpy_batch_from_dataframe(   batch_df,
                                                        col_id_x=self.col_id_x,
                                                        col_id_y=self.col_id_y,
                                                        root_dir=self.root_dir);
        
        if last_transform["horizontal_flip"]==True:
            X=DGMST.batch_multispectral_image_horizontal_flip(X);
            
        if last_transform["vertical_flip"]==True:
            X=DGMST.batch_multispectral_image_vertical_flip(X);
            
        if last_transform["zoom_factor"]!=1.0:
            X=DGMST.batch_multispectral_image_zoom(X, last_transform["zoom_factor"]);
        
        if last_transform["rotation_angle"]!=0.0:
            X=DGMST.batch_multispectral_image_rotate(X,last_transform["rotation_angle"]);
        
        #print('y batch',y)
        
        return X, y
