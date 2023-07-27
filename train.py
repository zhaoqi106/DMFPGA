from argparse import Namespace
from logging import Logger
import numpy as np
import os

import pandas as pd

from dmfpga.train import fold_train
from dmfpga.tool import set_log, set_train_argument, get_task_name, mkdir
import torch, gc

def training(args,log):


    info = log.info
    
    seed_first = args.seed
    data_path = args.data_path
    save_path = args.save_path
    
    score = []
    val_score = []
    train_score =[]
    
    for num_fold in range(args.num_folds):
        info(f'Seed {args.seed}')
        args.seed = seed_first + num_fold
        args.save_path = os.path.join(save_path,f'Seed_{args.seed}')
        mkdir(args.save_path)
        
        fold_score,fold_val_score,fold_train_score = fold_train(args,log)
        
        score.append(fold_score)
        val_score.append(fold_val_score)
        train_score.append(fold_train_score)
    score = np.array(score)
    val_score= np.array(val_score)
    train_score = np.array(train_score)
    
    info(f'Running {args.num_folds} folds in total.')
    if args.num_folds > 1:
        for num_fold, fold_score in enumerate(score):
           info(f'test {args.metric} = {np.nanmean(fold_score):.6f}')
           # info(f'Seed {seed_first + num_fold} : test {args.metric} = {np.nanmean(fold_score):.6f}')
    #score = np.nanmean(score, axis=1)
    score_ave = np.nanmean(score,axis=0)
    score_std = np.nanstd(score)
    val_score_ave = np.nanmean(val_score,axis=0)
    train_score_ave = np.nanmean(train_score,axis=0)

    info(f'Average train {args.metric} = {train_score_ave[3]:.6f}'
         f'  acc = {train_score_ave[0]:.6f}'
         f'  presion = {train_score_ave[1]:.6f}'
         f'  recall = {train_score_ave[2]:.6f}')


    info(f'Average val {args.metric} = {val_score_ave[4]:.6f}'
         f'  acc = {val_score_ave[0]:.6f}'
         f'  presion = {val_score_ave[1]:.6f}'
         f'  recall = {val_score_ave[2]:.6f}'
         f' spe = {val_score_ave[3]:.6f}')
    info(f' test {args.metric} = {score_ave[4]:.6f} ,'
         f'  acc = {score_ave[0]:.6f}'
         f'  presion = {score_ave[1]:.6f}'
         f'  recall = {score_ave[2]:.6f}'
         f'   spe = {score_ave[3]:.6f}')



    
    if args.task_num > 1:
        for i,one_name in enumerate(args.task_names):
            info(f'Average test {one_name} {args.metric} = {np.nanmean(score[:, i]):.6f} +/- {np.nanstd(score[:, i]):.6f}')
    
    return score_ave,score_std

if __name__ == '__main__':

    args = set_train_argument()
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    log = set_log('train',args.log_path)
    training(args,log)


    # e = []
    # args = set_train_argument()
    # log = set_log('train', args.log_path)
    # for i in range(20):
    #     np.random.seed(args.seed)
    #     args_tempt = args.seed
    #     torch.random.manual_seed(args.seed)
    #
    #     a, b = training(args, log)
    #     e.append([args.seed,a])
    #     args.seed = args_tempt + 1
    #     print(args.seed)
    # e = pd.DataFrame(columns=['seed', 'auc'], data=e)
    # e.to_csv(f'auc_score.csv')

