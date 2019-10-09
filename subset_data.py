import os
from src.pci_crackdown import * 

with open('data/output/training_data_tam.pkl' , 'rb') as f:
    df_train = pickle.load(f)
    
with open('data/output/testing_data_tam.pkl' , 'rb') as f:
    df_test = pickle.load(f)

def remove_obs(x, cutoff):
    to_drop = x['df']['days_since']  <= cutoff
    x['x'] = x['x'][-to_drop]
    x['y'] = x['y'][-to_drop]
    x['df'] = x['df'][to_drop == False]
    return x 

with open('data/output/training_data_tam_19890427.pkl' , 'wb') as f:
    pickle.dump(remove_obs(df_train,cutoff=1.5), f)
with open('data/output/testing_data_tam_19890427.pkl' , 'wb') as f:
    pickle.dump(remove_obs(df_test,cutoff=1.5), f)
