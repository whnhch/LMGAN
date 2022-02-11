from lmgan import Generator, Discriminator, Transformer, Classifier
from transformers import BertModel


from transformers import get_linear_schedule_with_warmup, AdamW
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import OrderedDict


import sys, os
cwd = os.getcwd()
sys.path.append(cwd)
from utils.utils import metric

def generate_optimizer(params, learning_rate, weight_decay, betas=(0.9, 0.999)):
    optimizer = AdamW(params,
                      lr=learning_rate,
                      betas=betas,
                      weight_decay=weight_decay,
                      eps=1e-8,)
    return optimizer


def generate_scheduler(optimizer, total_warmup_steps, total_train_steps):
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=total_warmup_steps,  # Default value in run_glue.py
                                                num_training_steps=total_train_steps,
                                                )
    return scheduler

def sample_lambda(lmbd, bs):
        alphas = []
        for i in range(bs):
            alpha = np.random.beta(lmbd, lmbd)
            alpha = min(alpha, 1.-alpha)
            alphas.append(alpha)
        alphas = np.asarray(alphas).reshape((bs,1))
        alphas = torch.from_numpy(alphas).float()
        alphas = alphas.cuda()
        
        return alphas

def criterion(probs, one_hot, smoothing=False, eps=0.1, num_labels=2):
    if smoothing:
        one_hot_ = one_hot * (1 - eps) + eps / num_labels +  (1 - one_hot) * eps / num_labels
        one_hot = one_hot_

    loss = -1*torch.mean(torch.sum(one_hot * torch.log(probs+1e-8), dim=-1), dim=0)

    return loss

bin_criterion=nn.BCELoss()


def train(transformer_config, l_train_dataloader,u_train_dataloader, g_l_train_dataloader, g_u_train_dataloader, dev_dataloader, \
    batch_size, epochs, warmup_proportion, device, num_labels, \
    z_size, g_hidden_size, g_dkp, x_size, d_hidden_size, d_dkp, weight_decay, max_len, label_list, alpha):

    generator1 = Generator(g_hidden_size, g_dkp).cuda()
    generator2 = Generator(g_hidden_size, g_dkp).cuda()
    generator3 = Generator(g_hidden_size, g_dkp).cuda()
    
    discriminator = Discriminator(x_size, d_hidden_size, d_dkp, num_labels).cuda()
    transformer = Transformer(transformer_config).cuda()
    
    bert = BertModel.from_pretrained('bert-base-uncased').cuda()
    classifier = Classifier(num_labels).cuda()

    g1_optimizer = generate_optimizer(generator1.parameters(), 5e-6, weight_decay)
    g2_optimizer = generate_optimizer(generator2.parameters(), 5e-6, weight_decay)
    g3_optimizer = generate_optimizer(generator3.parameters(), 5e-6, weight_decay)
    
    d_optimizer = generate_optimizer(discriminator.parameters(), 5e-6, weight_decay)
    t_optimizer = generate_optimizer(transformer.parameters(), 5e-6, weight_decay)
    
    bert_optimizer = generate_optimizer(bert.parameters(), 5e-5, weight_decay)
    cls_optimizer = generate_optimizer(classifier.parameters(), 5e-5, weight_decay)

    total_steps = 1000  * float(epochs)
    total_warmup_steps = int(total_steps * warmup_proportion)

    d_scheduler = generate_scheduler(d_optimizer, total_warmup_steps, total_steps)
    t_scheduler = generate_scheduler(t_optimizer, total_warmup_steps, total_steps)
    
    g1_scheduler = generate_scheduler(g1_optimizer, total_warmup_steps, total_steps)
    g2_scheduler = generate_scheduler(g2_optimizer, total_warmup_steps, total_steps)
    g3_scheduler = generate_scheduler(g3_optimizer, total_warmup_steps, total_steps)


    generator1 = nn.DataParallel(generator1)
    generator2 = nn.DataParallel(generator2)
    generator3 = nn.DataParallel(generator3)
    
    discriminator = nn.DataParallel(discriminator)
    transformer = nn.DataParallel(transformer)

    bert = nn.DataParallel(bert)
    classifier = nn.DataParallel(classifier)

    labeled_train_iter = iter(l_train_dataloader)
    unlabeled_train_iter = iter(u_train_dataloader)
    g_labeled_train_iter = iter(g_l_train_dataloader)
    g_unlabeled_train_iter = iter(g_u_train_dataloader)

    ce = nn.CrossEntropyLoss()
    
    # First fine-tuning BERT for generator.
    for epoch_i in range(0, epochs):
        for step in range(len(l_train_dataloader)):
            bert.train()
            classifier.train()
            try:
                    g_inputs_x, g_attention_x, g_gt_x = g_labeled_train_iter.next()
            except:
                    g_labeled_train_iter = iter(g_l_train_dataloader)
                    g_inputs_x, g_attention_x, g_gt_x = g_labeled_train_iter.next()

            g_inputs_x, g_attention_x,g_gt_x = g_inputs_x.cuda(), g_attention_x.cuda(), g_gt_x.cuda()
            z = bert(input_ids=g_inputs_x, attention_mask=g_attention_x, return_dict=True, output_hidden_states='True')

            pooling = z['hidden_states'][-1][:,0,:]
            x, logits, probs = classifier(pooling)
            
            loss = ce(probs, g_gt_x-1)

            cls_optimizer.zero_grad()
            bert_optimizer.zero_grad()

            loss.backward()
            cls_optimizer.step()
            bert_optimizer.step()
    
    classifier.eval()

    # Sencond semi-supervised learning.  
    for epoch_i in range(0,epochs):
        generator1.train()
        generator2.train()
        generator3.train()
        
        transformer.train()
        discriminator.train()

        bert.eval()

        for step in range(1000):
            transformer.train()
            generator1.train()
            generator2.train()
            generator3.train()
            discriminator.train()
            bert.eval()

            try:
                inputs_x, attention_x, gt_x = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(l_train_dataloader)
                inputs_x, attention_x, gt_x = labeled_train_iter.next()
            try:
                u_inputs_x, u_attention_x, u_gt_x= unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(u_train_dataloader)
                u_inputs_x, u_attention_x, u_gt_x = unlabeled_train_iter.next()
            batch_size_ = len(inputs_x) + len(u_inputs_x)
            
            try:
                g_inputs_x, g_attention_x, g_gt_x = g_labeled_train_iter.next()
            except:
                g_labeled_train_iter = iter(g_l_train_dataloader)
                g_inputs_x, g_attention_x, g_gt_x = g_labeled_train_iter.next()
            try:
                g_u_inputs_x, g_u_attention_x, g_u_gt_x = g_unlabeled_train_iter.next()
            except:
                g_unlabeled_train_iter = iter(g_u_train_dataloader)
                g_u_inputs_x, g_u_attention_x, g_u_gt_x = g_unlabeled_train_iter.next()
            batch_size_ = len(inputs_x) + len(u_inputs_x)

            inputs_x, attention_x, gt_x = inputs_x.cuda(), attention_x.cuda(), gt_x.cuda()
            u_inputs_x, u_attention_x, u_gt_x = u_inputs_x.cuda(), u_attention_x.cuda(), u_gt_x.cuda()

            g_inputs_x, g_attention_x,g_gt_x = g_inputs_x.cuda(), g_attention_x.cuda(), g_gt_x.cuda()
            g_u_inputs_x, g_u_attention_x, g_u_gt_x = g_u_inputs_x.cuda(), g_u_attention_x.cuda(), g_u_gt_x.cuda()

            labeled_example_cnt = len(inputs_x)
            unlabeled_example_cnt = len(u_inputs_x)
            b_input_ids = torch.cat((inputs_x, u_inputs_x), dim=0)
            b_input_masks = torch.cat((attention_x, u_attention_x), dim=0)
            b_gts = torch.cat((gt_x, u_gt_x), dim=0)

            g_input_ids = torch.cat((g_inputs_x, g_u_inputs_x), dim=0)
            g_input_masks = torch.cat((g_attention_x, g_u_attention_x), dim=0)
            g_gts = torch.cat((g_gt_x, g_u_gt_x), dim=0)

            label_indicies = torch.zeros(len(b_input_ids),dtype=torch.bool)
            label_indicies[:len(inputs_x)] = True
            unlabel_indicies = torch.zeros(len(b_input_ids),dtype=torch.bool)
            unlabel_indicies[len(inputs_x):] = True

            # Get fake data for updating Discriiminator
            z = bert(input_ids=g_input_ids, attention_mask=g_input_masks, return_dict=True, output_hidden_states='True')
            z12 = z['hidden_states'][-1].detach()
            z10 = z['hidden_states'][-3].detach()
            z9= z['hidden_states'][-4].detach()
            
            z12_ = torch.randn([len(z12), max_len, 768], requires_grad=False).to(device)
            z10_ = torch.randn([len(z9), max_len, 768], requires_grad=False).to(device)
            z9_ = torch.randn([len(z9), max_len, 768], requires_grad=False).to(device)

            z12 += z12_
            z10 += z10_
            z9 += z9_

            g_hidden_layer1 = generator1(z12)
            g_hidden_layer2 = generator2(z10)
            g_hidden_layer3 = generator3(z9)
            g_hidden_layer = torch.cat((g_hidden_layer1.detach(), g_hidden_layer2.detach(), g_hidden_layer3.detach()))
            d_fake_features, d_fake_logits, d_fake_prob = discriminator(g_hidden_layer.detach())

            pool = transformer(input_ids=b_input_ids, input_masks=b_input_masks)

            #######################################################################
            # Mixup REAL and FAKE
            #######################################################################

            # get distribution of real data for pseudo labeling. 
            with torch.no_grad():
                pseudo_feature, pseudo_logits, pseudo_prob = discriminator(pool.detach())

            # pseudo labeling real data
            b_one_hot_labels = nn.functional.one_hot(b_gts, num_classes=num_labels)[:,1:] # excepting first label which is unsup label
            pseudo_prob = nn.functional.softmax(pseudo_logits[:,1:],dim=-1)
            b_one_hot_labels = b_one_hot_labels.float()
            b_one_hot_labels[unlabel_indicies] = pseudo_prob[unlabel_indicies]

            g_one_hot_labels = nn.functional.one_hot(g_gts, num_classes=num_labels)[:,1:] # excepting first label which is unsup label
            d_fake_prob_= d_fake_logits.reshape(len(b_gts),3,num_labels)[:,:,1:]
            d_fake_prob_= nn.functional.softmax(d_fake_prob_,dim=-1)

            # average the distributions
            d_fake_prob_ = (d_fake_prob_[:,0,:] + d_fake_prob_[:,1,:] + d_fake_prob_[:,2,:])/3
            g_one_hot_labels = g_one_hot_labels.float()
            g_one_hot_labels[unlabel_indicies,:] = d_fake_prob_[unlabel_indicies,:]


            lmbd = sample_lambda(alpha, batch_size)

            random_idx = torch.randint(0,batch_size_-1, label_indicies.shape)

            mixup1 = lmbd*pool + (1-lmbd)*g_hidden_layer1[random_idx]
            mixup2 = lmbd*pool + (1-lmbd)*g_hidden_layer2[random_idx]
            mixup3 = lmbd*pool + (1-lmbd)*g_hidden_layer3[random_idx]
            mixup = torch.cat((mixup1, mixup2, mixup3),dim=0)

            mixup_labels = lmbd*b_one_hot_labels + (1-lmbd)*g_one_hot_labels[random_idx]
            mixup_labels = torch.cat((mixup_labels, mixup_labels, mixup_labels), dim=0)

            ## output of Mixup REAL And FAKE
            real_features, real_logits, real_prob = discriminator(mixup)
            
            real_logits = real_logits[:, 1:]  # except fake label # B
            real_probs = nn.functional.softmax(real_logits, dim=-1)

            mixup_label_indicies = torch.cat((label_indicies, label_indicies, label_indicies),dim=0)
            mixup_unlabel_indicies= torch.cat((unlabel_indicies, unlabel_indicies, unlabel_indicies),dim=0)

            # supervised learning loss
            d_l_supervised = criterion(real_probs[mixup_label_indicies], mixup_labels[mixup_label_indicies], smoothing=False, eps=0.01, num_labels=len(mixup_labels[0])-1)
            # kl divergence
            d_l_supervised += torch.nn.functional.kl_div(real_probs[mixup_unlabel_indicies].log(), mixup_labels[mixup_unlabel_indicies], None, None, 'batchmean')

            # mixup loss
            mixed_prob_ = torch.unsqueeze(real_prob[:,0],dim=-1)
            unsup_prob = torch.cat((mixed_prob_, 1- mixed_prob_),dim=1)
            unsup_label = 1-lmbd
            unsup_label = torch.cat((unsup_label, 1-unsup_label), dim=1)
            unsup_label = torch.cat((unsup_label, unsup_label, unsup_label), dim=0)
            d_l_unsupervised = criterion(unsup_prob, unsup_label, smoothing=False, eps=0.1)
           
            # regularizations
            p = torch.ones(num_labels-1).cuda() / (num_labels-1)

            prob_avg = torch.mean(real_probs, dim=0)
            L_p_1 = -torch.sum(torch.log(prob_avg) * p)
            L_e_1 = -torch.mean(torch.sum(real_probs * nn.functional.log_softmax(real_logits, dim=1), dim=1))

            d_fake_prob_ = torch.unsqueeze(d_fake_prob[:,0],dim=-1)
            d_fake_prob_ = torch.cat((d_fake_prob_, 1- d_fake_prob_),dim=1)
            d_l_unsupervised_g = criterion(d_fake_prob_, 1-unsup_label, smoothing=False, eps=0.1)

            #######################################################################
            # Mixup Real and Real
            #######################################################################

            mixup = lmbd*pool + (1-lmbd)*pool[random_idx]
            real_features2, real_logits2, real_prob2 = discriminator(mixup)
            
            b_one_hot_labels = nn.functional.one_hot(b_gts, num_classes=num_labels)[:,1:]
            b_one_hot_labels = b_one_hot_labels.float()
            b_one_hot_labels[unlabel_indicies] = pseudo_prob[unlabel_indicies]
            
            mixup_label = lmbd*b_one_hot_labels + (1-lmbd)*b_one_hot_labels[random_idx]
            
            real_probs =  nn.functional.softmax(real_logits2[:,1:], dim=-1)
            d_l_supervised2 = criterion(real_probs[label_indicies], mixup_label[label_indicies], smoothing=False, eps=0.01, num_labels=len(mixup_label[0])-1)
            d_l_supervised2 += torch.nn.functional.kl_div(real_probs[unlabel_indicies].log(), mixup_label[unlabel_indicies], None, None, 'batchmean')
            
            p = torch.ones(num_labels-1).cuda() / (num_labels-1)

            prob_avg = torch.mean(real_probs, dim=0)
            L_p_2 = -torch.sum(torch.log(prob_avg) * p)
            L_e_2 = -torch.mean(torch.sum(real_probs * nn.functional.log_softmax(real_logits2[:,1:], dim=1), dim=1))

            mixed_prob_ = torch.unsqueeze(real_prob2[:,0],dim=-1)
            unsup_prob = torch.cat((mixed_prob_, 1- mixed_prob_),dim=1)
            unsup_label = torch.full( (len(b_gts),1), 0, dtype=torch.float).cuda()
            unsup_label = torch.cat((unsup_label, 1-unsup_label), dim=1)
            d_l_unsupervised2 = criterion(unsup_prob, unsup_label, smoothing=False, eps=0.1)    


            #######################################################################
            # Mixup Fake and Fake
            #######################################################################

            mixup1 = lmbd*g_hidden_layer1 + (1-lmbd)*g_hidden_layer2[random_idx]
            mixup2 = lmbd*g_hidden_layer2 + (1-lmbd)*g_hidden_layer3[random_idx]
            mixup3 = lmbd*g_hidden_layer3 + (1-lmbd)*g_hidden_layer1[random_idx]
            mixup = torch.cat((mixup1, mixup2, mixup3),dim=0)

            d_fake_features2, d_fake_logits2, d_fake_prob2 = discriminator(mixup)

            d_fake_prob_ = torch.unsqueeze(d_fake_prob2[:,0],dim=-1)
            g_prob = torch.cat((d_fake_prob_, 1-d_fake_prob_),dim=1)
            g_label = torch.full( (len(g_gts),1), 1, dtype=torch.float).cuda()
            g_label = torch.cat((g_label, 1-g_label), dim=1)
            g_label = torch.cat((g_label, g_label, g_label), dim=0)
            d_l_unsupervised_g2 = criterion(g_prob, g_label, smoothing=False)
            
            d_loss = 0.8*(L_p_1 + L_p_2) + 0.4*(L_e_1 + L_e_2) +\
                 d_l_supervised +  + d_l_unsupervised + d_l_unsupervised_g +\
                 d_l_supervised2 + d_l_unsupervised_g2 + d_l_unsupervised2  



            #######################################################################
            # Generator
            #######################################################################

            z = bert(input_ids=g_input_ids, attention_mask=g_input_masks, return_dict=True, output_hidden_states='True')
            z12 = z['hidden_states'][-1].detach()
            z10 = z['hidden_states'][-3].detach()
            z9 = z['hidden_states'][-4].detach()
        
            z12_ = torch.randn([len(z12), max_len, 768], requires_grad=False).to(device)
            z10_ = torch.randn([len(z10), max_len, 768], requires_grad=False).to(device)
            z9_ = torch.randn([len(z9), max_len, 768], requires_grad=False).to(device)

            z12 += z12_
            z10 += z10_
            z9 += z9_

            g_hidden_layer1 = generator1(z12)
            g_hidden_layer2 = generator2(z10)
            g_hidden_layer3 = generator3(z9)
            g_hidden_layer = torch.cat((g_hidden_layer1, g_hidden_layer2, g_hidden_layer3))


            #######################################################################
            # Mixup of FAKE and REAL
            #######################################################################
            lmbd = sample_lambda(alpha, batch_size)
            mixup1 = lmbd*g_hidden_layer1 + (1-lmbd)*pool[random_idx]
            mixup2 = lmbd*g_hidden_layer2 + (1-lmbd)*pool[random_idx]
            mixup3 = lmbd*g_hidden_layer3 + (1-lmbd)*pool[random_idx]
            mixup = torch.cat((mixup1, mixup2, mixup3),dim=0)

            with torch.no_grad():
                pseudo_feature, pseudo_logits, pseudo_prob = discriminator(pool.detach())
                f_pseudo_feature, f_pseudo_logits, f_pseudo_prob = discriminator(g_hidden_layer.detach())

            b_one_hot_labels = nn.functional.one_hot(b_gts, num_classes=num_labels)[:,1:] # excepting first label which is unsup label
            pseudo_prob_ = nn.functional.softmax(pseudo_logits[:,1:],dim=-1)            
            b_one_hot_labels = b_one_hot_labels.float()
            b_one_hot_labels[unlabel_indicies] = pseudo_prob_[unlabel_indicies]
        
            f_pseudo_logits=f_pseudo_logits.reshape((len(b_gts),3,num_labels))
            f_pseudo_prob= nn.functional.softmax(f_pseudo_logits[:,:,1:], dim=-1)
            
            g_one_hot_labels = nn.functional.one_hot(g_gts, num_classes=num_labels)[:,1:]

            f_pseudo_prob = (f_pseudo_prob[:,0,:] + f_pseudo_prob[:,1,:] + f_pseudo_prob[:,2,:])/3
            g_one_hot_labels = g_one_hot_labels.float()
            g_one_hot_labels[unlabel_indicies,:] = f_pseudo_prob[unlabel_indicies,:]
            
            mixup_labels = lmbd*g_one_hot_labels + (1-lmbd)*b_one_hot_labels[random_idx]
            
            mixup_labels = torch.cat((mixup_labels, mixup_labels, mixup_labels), dim=0)
            
            d_fake_features, d_fake_logits, d_fake_prob = discriminator(mixup)

            # L_fm
            g_feature = torch.mean(torch.square(torch.mean(real_features) - torch.mean(d_fake_features)))

            d_fake_prob_ = torch.unsqueeze(d_fake_prob[:,0],dim=-1)
            g_prob = torch.cat((d_fake_prob_, 1-d_fake_prob_),dim=1)
            g_label = 1-lmbd
            g_label = torch.cat((g_label, 1-g_label), dim=1)
            g_label = torch.cat((g_label, g_label, g_label), dim=0)
            g_l_unsup1 = criterion(g_prob, g_label, smoothing=False)

            fake_logits = d_fake_logits[:, 1:]  # except fake label # B
            fake_probs = nn.functional.softmax(fake_logits, dim=-1)
            mixup_label_indicies= torch.cat((label_indicies, label_indicies, label_indicies),dim=0)
            mixup_unlabel_indicies= torch.cat((unlabel_indicies, unlabel_indicies, unlabel_indicies),dim=0)

            g_l_supervised1 = criterion(fake_probs[mixup_label_indicies], mixup_labels[mixup_label_indicies], smoothing=False, eps=0.01, num_labels=len(mixup_labels[0])-1)
            g_l_supervised1 += torch.nn.functional.kl_div(fake_probs[mixup_unlabel_indicies].log(), mixup_labels[mixup_unlabel_indicies], None, None, 'batchmean')

            p = torch.ones(num_labels-1).cuda() / (num_labels-1)
            prob_avg = torch.mean(fake_probs, dim=0)
            L_p_1 = -torch.sum(torch.log(fake_probs) * p)
            L_e_1 = -torch.mean(torch.sum(fake_probs * nn.functional.log_softmax(fake_logits, dim=1), dim=1))

            #######################################################################
            # Mixup of FAKE and FAKE
            #######################################################################
            mixup1 = lmbd*g_hidden_layer1 + (1-lmbd)*g_hidden_layer2[random_idx]
            mixup2 = lmbd*g_hidden_layer2 + (1-lmbd)*g_hidden_layer3[random_idx]
            mixup3 = lmbd*g_hidden_layer3 + (1-lmbd)*g_hidden_layer1[random_idx]
            mixup = torch.cat((mixup1, mixup2, mixup3),dim=0)

            g_one_hot_labels = nn.functional.one_hot(g_gts, num_classes=num_labels)[:,1:] # excepting first label which is unsup label
            
            d_fake_prob=d_fake_logits.reshape((len(b_gts),3,num_labels))[:,:,1:]
            d_fake_prob=nn.functional.softmax(d_fake_prob,dim=-1)

            d_fake_prob_ = (d_fake_prob[:,0,:] + d_fake_prob[:,1,:] + d_fake_prob[:,2,:])/3
            g_one_hot_labels = g_one_hot_labels.float()
            g_one_hot_labels[unlabel_indicies,:] = d_fake_prob_[unlabel_indicies,:]

            mixup_labels = lmbd*g_one_hot_labels + (1-lmbd)*g_one_hot_labels[random_idx]

            mixup_labels = torch.cat((mixup_labels, mixup_labels, mixup_labels), dim=0)

            d_fake_features2, d_fake_logits2, d_fake_prob2 = discriminator(mixup)
            fake_logits = d_fake_logits2[:, 1:]  # except fake label # B
            fake_probs = nn.functional.softmax(fake_logits, dim=-1)
            g_l_supervised2 = criterion(fake_probs[mixup_label_indicies], mixup_labels[mixup_label_indicies], smoothing=False, eps=0.01, num_labels=len(mixup_labels[0])-1)
            g_l_supervised2 += torch.nn.functional.kl_div(fake_probs[mixup_unlabel_indicies].log(), mixup_labels[mixup_unlabel_indicies], None, None, 'batchmean')

            p = torch.ones(num_labels-1).cuda() / (num_labels-1)
            prob_avg = torch.mean(fake_probs, dim=0)
            L_p_2 = -torch.sum(torch.log(fake_probs) * p)
            L_e_2 = -torch.mean(torch.sum(fake_probs * nn.functional.log_softmax(fake_logits, dim=1), dim=1))

            d_fake_prob_ = torch.unsqueeze(d_fake_prob2[:,0],dim=-1)
            g_prob = torch.cat((d_fake_prob_, 1-d_fake_prob_),dim=1)
            g_label = torch.full( (len(g_gts),1), 0, dtype=torch.float).cuda()
            g_label = torch.cat((g_label, 1-g_label), dim=1)
            g_label = torch.cat((g_label, g_label, g_label), dim=0)
            g_l_unsup2 = criterion(g_prob, g_label, smoothing=False)

            
            g_loss = 0.8*(L_p_1 + L_p_2) + 0.4*(L_e_1 + L_e_2) + g_feature + g_l_supervised1 +  g_l_unsup1 + g_l_supervised2 + g_l_unsup2



            g1_optimizer.zero_grad()
            g2_optimizer.zero_grad()
            g3_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)

            g1_optimizer.step()
            g1_scheduler.step()

            g2_optimizer.step()
            g2_scheduler.step()

            g3_optimizer.step()
            g3_scheduler.step()

            d_optimizer.zero_grad()
            t_optimizer.zero_grad()

            d_loss.backward()

            d_optimizer.step()
            d_scheduler.step()
            
            t_optimizer.step()
            t_scheduler.step()

            if step%30 == 0 and step != 0:
                        predictions = []
                        true_labels = []

                        total_loss = []

                        discriminator.eval()
                        transformer.eval()

                        for step2, batch2 in enumerate(dev_dataloader):
                            batch2 = tuple(t.cuda() for t in batch2)

                            b_input_ids, b_input_masks, b_label_ids = batch2

                            with torch.no_grad():
                                # Forward pass, calculate logit predictions
                                pool = transformer(input_ids=b_input_ids, input_masks=b_input_masks)

                                d_real_features, d_real_logits, d_real_prob = discriminator(x=pool)

                                logits = d_real_logits[:,1:]
                                log_probs = torch.log_softmax(logits, dim=-1)

                                one_hot_labels = nn.functional.one_hot(b_label_ids, num_classes=num_labels)[:,1:]
                                loss = -torch.mean(torch.sum(one_hot_labels * log_probs, dim=-1))

                            probs = log_probs.detach().cpu().numpy()
                            loss = loss.detach().cpu().numpy()

                            total_loss.append(loss)

                            label_ids = b_label_ids.to('cpu').numpy()
                            predictions.append(np.argmax(probs, axis=1))
                            true_labels.append(label_ids-1)
                        y_pred, y_true = [], []
                        loss = []
                        row = len(predictions)
                        for i in range(row):
                            col = len(predictions[i])
                            for j in range(col):
                                tmp1 = predictions[i][j]
                                tmp2 = true_labels[i][j]
                                y_pred.append(label_list[int(tmp1)])
                                y_true.append(label_list[int(tmp2)])
                            loss.append(total_loss[i])

                        accuracy, precision, recall, f1_micro, f1_macro = metric(y_pred, y_true)
                        print('[{:}/1000]'.format(step))
                        print('loss : {:}'.format(sum(loss) / len(loss)))
                        cur_loss =sum(loss) / len(loss)
                        print(
                            'accuracy : {:},\tprecision : {:},\trecall : {:},\tf1_micro : {:},\tf1_macro : {:}'.format(accuracy, precision,
                                                                                                                    recall, f1_micro,
                                                                                                                f1_macro))
        
        torch.save({
		                'epoch': epoch_i,
		                'gen1_state_dict': generator1.state_dict(),
		                'gen2_state_dict': generator2.state_dict(),
		                'gen3_state_dict': generator3.state_dict(),
		                'dis_state_dict': discriminator.state_dict(),
		                'trs_state_dict': transformer.state_dict(),
		                'bert_state_dict': bert.state_dict(),
		                'g1_optimizer_state_dict': g1_optimizer.state_dict(),
		                'g2_optimizer_state_dict': g2_optimizer.state_dict(),
		                'g3_optimizer_state_dict': g3_optimizer.state_dict(),
		                'd_optimizer_state_dict': d_optimizer.state_dict(),
		                'd_scheduler_state_dict':d_scheduler.state_dict(),
		                'g1_scheduler_state_dict':g1_scheduler.state_dict(),
		                'g2_scheduler_state_dict':g2_scheduler.state_dict(),
		                'g3_scheduler_state_dict':g3_scheduler.state_dict(),
		}, 'lmgan_' + str(epoch_i) + '.pt')
        

def evaluate(transformer_config, chkpt, device, test_dataloader, label_list,\
             x_size, d_hidden_size, d_dkp):
    print('evaluate test set')
    num_labels = len(label_list)

    discriminator = Discriminator(x_size, d_hidden_size, d_dkp, num_labels)
    transformer = Transformer(transformer_config)
        
    try:
        discriminator.load_state_dict(chkpt['dis_state_dict'])
        transformer.load_state_dict(chkpt['trs_state_dict'])
    except:
        new_dis_state_dict = OrderedDict()
        for k, v in chkpt['dis_state_dict'].items():
            name = k[7:] # remove module.
            new_dis_state_dict[name] = v

        new_trs_state_dict = OrderedDict()
        for k, v in chkpt['trs_state_dict'].items():
            name = k[7:] # remove module.
            new_trs_state_dict[name] = v
        discriminator.load_state_dict(new_dis_state_dict)
        transformer.load_state_dict(new_trs_state_dict)

    predictions = []
    true_labels = []

    total_loss = []

    discriminator.to(device)
    transformer.to(device)

    discriminator.eval()
    transformer.eval()

    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_masks, b_label_ids = batch

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            pool = transformer(input_ids=b_input_ids, input_masks=b_input_masks)

            d_real_features, d_real_logits, d_real_prob = discriminator(x=pool)

            logits = d_real_logits[:,1:]
            log_probs = torch.log_softmax(logits, dim=-1)

            one_hot_labels = nn.functional.one_hot(b_label_ids, num_classes=num_labels)[:,1:]
            loss = -torch.mean(torch.sum(one_hot_labels * log_probs, dim=-1))

        probs = log_probs.detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()

        total_loss.append(loss)

        label_ids = b_label_ids.to('cpu').numpy()
        predictions.append(np.argmax(probs, axis=1))
        true_labels.append(label_ids-1)

    y_pred, y_true = [], []
    loss = []
    row = len(predictions)
    for i in range(row):
        col = len(predictions[i])
        for j in range(col):
            tmp1 = predictions[i][j]
            tmp2 = true_labels[i][j]
            y_pred.append(label_list[int(tmp1)])
            y_true.append(label_list[int(tmp2)])
        loss.append(total_loss[i])

    accuracy, precision, recall, f1_micro, f1_macro = metric(y_pred, y_true)

    print('loss : {:}'.format(sum(loss) / len(loss)))
    print(
        'accuracy : {:},\tprecision : {:},\trecall : {:},\tf1_micro : {:},\tf1_macro : {:}'.format(accuracy, precision,
                                                                                                   recall, f1_micro,
                                                                                                   f1_macro))

    df_result = pd.DataFrame(y_pred)
    df_result.to_csv('./dataset/'+'lmgan_test_pred.csv', header=None, index=False)

    df_result = pd.DataFrame(y_true)
    df_result.to_csv('./dataset/'+'lmgan_test_gt.csv', header=None, index=False)

    return accuracy, f1_micro, f1_macro 
