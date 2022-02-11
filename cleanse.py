from  utils.features import *
import pandas as pd
import numpy as np
import argparse


def ag_split_data(dataset_dir, label_num=0, unlabel_num=0):
    seed_val = 100
    np.random.seed(seed_val)
    with open(dataset_dir+'train.csv') as f1, open(dataset_dir+'test.csv') as f2:
        df = pd.read_csv(f1, index_col=None, sep=',', header=None)
        test = np.array(pd.read_csv(f2, index_col=None, sep=',', header=None))

    array = np.array(df)

    
    mixtext_test_num=1900
    classes_num = len(AGProcessor().get_labels())-1

    labeled_idxs = []
    unlabeled_idxs = []
    dev_idxs=[]
    test_=[]

    for i in range(1,classes_num+1):
        idxs = np.where(array[:,0]==i)[0]
        np.random.shuffle(idxs)
        
        labeled_idxs.extend(idxs[:int(label_num)])
        unlabeled_idxs.extend(idxs[int(label_num):int(label_num)+unlabel_num])
        dev_idxs.extend(idxs[-2000:])
        
        clss_arr = test[test[:,0]==i]
        test_.append(clss_arr[:int(mixtext_test_num),:]) 

    labeled_df = pd.DataFrame(array[labeled_idxs])
    labeled_df.to_csv(dataset_dir+'evenly_labeled_'+str(label_num)+'.csv', header=None, index=False)

    unlabeled_df = pd.DataFrame(array[unlabeled_idxs])
    unlabeled_df.to_csv(dataset_dir+'evenly_unlabeled_'+str(label_num)+'.csv', header=None, index=False)

    dev = pd.DataFrame(array[dev_idxs])
    dev.to_csv(dataset_dir+'evenly_dev_'+str(label_num)+'.csv', header=None, index=False)

    test_ = np.array(test_).reshape(-1,3)
    test_df = pd.DataFrame(test_)
    test_df.to_csv(dataset_dir+'evenly_test.csv', header=None, index=False)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_num', type=float, default=10)
    parser.add_argument('--unlabel_num', type=int, default=5000)
    parser.add_argument('--dataset_dir', type=str, default='dataset/')

    args = parser.parse_args()

    ag_split_data(args.dataset_dir, args.label_num, args.unlabel_num)