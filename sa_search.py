import os
from src.pci_crackdown import * 

with open('data/output/training_data_tam.pkl' , 'rb') as f:
    df_train = pickle.load(f)

with open('data/output/testing_data_tam.pkl' , 'rb') as f:
    df_test = pickle.load(f)


if not os.path.exists('Results/models/best.model'):
    tamhk = pci_crackdown()
    tamhk.run(df_train, df_test)
    tamhk.save('Results/models/best')
    print(tamhk.loss)


tamhk = pci_crackdown.load('Results/models/best')
print(tamhk.loss)

## GPU could run out of memory and crash if period is large
sa = tamhk.sa(df_train, df_test, T=0.05, discount=0.05, bandwidth = 0.05, period = 10)

if sa is not None:
    print('*************')
    print('better model!')
    print('*************')
    sa.save('model/best')

    print(tamhk.loss)
    print(sa.loss)
