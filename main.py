from transformers import BertConfig, BertTokenizer
from utils.tokenization import getTokens
import train_lmgan

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import random
import numpy as np
import argparse

from utils import module_dict

g_dkp = 0.9

d_dkp = 0.9
x_size = 768
z_size = 100
d_hidden_size = 768
g_hidden_size = 768

weight_decay = 0
warmup_proportion = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(is_training, is_evaluating, label_rate, dataset_dir, max_length, data_processor, \
    seed_val, epochs, batch_size, alpha, b_num_label):
    
    model_name = 'lmgan'
    
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    print(model_name)
    processor=module_dict.processor[data_processor.lower()]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-uncased')
    transformer_config = BertConfig.from_pretrained(pretrained_model_name_or_path='bert-base-uncased')

    if is_training:
        labeled_examples = []
        labeled_examples = processor.create_examples(dataset_dir+'labeled_'+str(label_rate)+'.csv', is_unsup=False)  # dict

        unlabeled_examples = []
        unlabeled_examples = processor.create_examples(dataset_dir+'unlabeled_'+str(label_rate)+'.csv', is_unsup=True)

        dev_examples = []
        dev_examples = processor.create_examples(dataset_dir+'dev_'+str(label_rate)+'.csv', is_unsup=False)
       
        l_tr_input_ids, l_tr_attention_masks, l_tr_gts = getTokens(tokenizer,labeled_examples,processor,
                                                                                        max_length,
                                                                                        1)

        u_tr_input_ids, u_tr_attention_masks, u_tr_gts = getTokens(tokenizer,unlabeled_examples,processor,
                                                                                        max_length,
                                                                                        1)
            
        l_train_dataset = TensorDataset(l_tr_input_ids, l_tr_attention_masks, l_tr_gts)
        u_train_dataset = TensorDataset(u_tr_input_ids, u_tr_attention_masks, u_tr_gts)

        l_train_sampler = RandomSampler(l_train_dataset)
        l_train_dataloader = DataLoader(l_train_dataset, sampler=l_train_sampler, batch_size=b_num_label)
        
        u_train_sampler = RandomSampler(u_train_dataset)
        u_train_dataloader = DataLoader(u_train_dataset, sampler=u_train_sampler, batch_size=batch_size-b_num_label)


        g_l_train_sampler = RandomSampler(l_train_dataset)
        g_l_train_dataloader = DataLoader(l_train_dataset, sampler=g_l_train_sampler, batch_size=b_num_label)
        
        g_u_train_sampler = RandomSampler(u_train_dataset)
        g_u_train_dataloader = DataLoader(u_train_dataset, sampler=g_u_train_sampler, batch_size=batch_size-b_num_label)

        
        dev_input_ids, dev_attention_masks, dev_gts = getTokens(tokenizer, dev_examples, processor, max_length,
                                                                                        1)
        dev_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_gts)
        dev_sampler = RandomSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=batch_size)


        print('start learning LMGAN')
        train_lmgan.train(transformer_config, l_train_dataloader, u_train_dataloader,  g_l_train_dataloader, g_u_train_dataloader, dev_dataloader,\
                    batch_size, epochs, warmup_proportion, device, len(label_list),\
                         z_size, g_hidden_size, g_dkp, x_size, d_hidden_size, d_dkp, weight_decay, max_length, label_list, alpha)  

    if is_evaluating:
        test_examples = processor.create_examples(dataset_dir+'test'+'.csv', is_unsup=False)  # dict

        test_input_ids, test_attention_masks, test_gts = getTokens(tokenizer,test_examples,
                                                                                        processor,
                                                                                        max_length,
                                                                                        factor=1)

        test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_gts)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=8)

        chkpt_name = 'lmgan_'+str(epochs-1)+'.pt'
        print(chkpt_name)
        chkpt = torch.load(chkpt_name)
        
        print('start evaluating LMGAN')
        train_lmgan.evaluate(transformer_config, chkpt, device, test_dataloader, label_list, x_size, d_hidden_size, d_dkp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_num', type=float, default=1.0)
    parser.add_argument('--max_length', type=int, default=64)

    parser.add_argument('--is_training', type= str2bool, default=True)
    parser.add_argument('--is_evaluating', type= str2bool, default=True)

    parser.add_argument('--dataset_dir', type= str, default='dataset/')
    parser.add_argument('--data_processor', type= str, default='sst5')

    parser.add_argument('--seed_val', type= int, default=100)

    parser.add_argument('--epochs', type= int, default=3)
    parser.add_argument('--batch_size', type= int, default=32)

    parser.add_argument('--alpha',type=float, default=0.5)

    parser.add_argument('--b_num_label',type=int, default=2)

    args = parser.parse_args()

    main(args.is_training, args.is_evaluating, args.label_num, args.dataset_dir, \
        args.max_length, args.data_processor, args.seed_val, args.epochs, args.batch_size, args.alpha, args.b_num_label)
