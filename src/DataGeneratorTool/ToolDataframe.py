import numpy as np
import pandas as pd
def repeat_and_balance_data_in_dataframe(df,col_id_y,label_list):
    '''
    
    Parameters
    
    df: dataframe
    col_id_y: int : Identificator of column value with y sample. 
                    This column will be rounded and compared with label_list.
    label_list:list: List of labels to be balanced in your quantity.
    
    '''
    
    y=np.round(df.iloc[:,col_id_y].to_numpy());
    y_list=y.tolist();

    w=[];
    for val in label_list:
        w.append(y_list.count(val))

    MAX=np.max(w);

    DataFrame=[];
    for val in label_list:
        # IDs con val
        IDS=(y==val).nonzero()[0].tolist();

        # dataframe com IDs
        DF=df.iloc[IDS,:];
        L=DF.shape[0];

        # ids a ser repetidos
        ids=np.random.randint(0,L-1, size=(MAX-L))
        if len(ids)>0:
            DataFrame.append(DF.iloc[ids.tolist(),:]);
    return pd.concat(DataFrame);

