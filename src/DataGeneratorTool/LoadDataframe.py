import numpy as np
import os

def load_numpy_batch_from_dataframe(df,col_id_x=0,col_id_y=1,root_dir=None):
    '''
    Load a batch of samples (X,y) from a pandas dataframe.
    
    Parameters
    
    df: Pandas dataframe
    col_id_x: int : Identificator of column string with X sample.
    col_id_y: int : Identificator of column value with y sample.
    root_dir: str : Root path of filepath in the column_id_x.
    
    Return
    
    X,y: X is an 3D numpy array, y is an 1D numpy array.
    
    '''
    filename=df.iloc[:,col_id_x].to_numpy();
    y=df.iloc[:,col_id_y].to_numpy();
    X=[];

    for n in range(np.size(filename)):
        if isinstance(root_dir, str):
            X.append(np.load(os.path.join(root_dir,filename[n])));
        else:
            X.append(np.load(filename[n]));
    X=np.stack(X);
    
    return X,y
