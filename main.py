import os
import torch
import numpy as np
import pandas as pd
import argparse
import scipy.sparse as ss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from torch.utils.tensorboard import SummaryWriter


from MFmodel import MFModel
import loadData
import utils

if __name__ == "__main__":
    #set seed
    np.random.seed(42)
    torch.manual_seed(42)

    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="yahoo")  # dataset
    # only useful for KuaiRand
    parser.add_argument("--key", default="is_click")
    parser.add_argument("--binary", action='store_true')  # false if not given

    # dims of latent factor
    parser.add_argument("--num", default=50, type=int)
    parser.add_argument("--iter", default=30, type=int)  # max iteration
    parser.add_argument("--batch", default=1024, type=int)  # batch size
    parser.add_argument("--lr", default=1e-3, type=float)  # learning rate
    parser.add_argument("--l2", default=1e-4, type=float)  # L2 regularization
    parser.add_argument("--verbose", action='store_true')  # false if not given
    parser.add_argument("--early_stop", action='store_true')
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--export", action='store_true')
    args = parser.parse_args()
    #args = parser.parse_args('--dataset=coat --num=10 --iter=2 --lr=1e-3 --l2=1e-5 --export --binary'.split())
    #print(args)

    if not args.binary:
        save_dir = 'model/raw_%s_num%d_batch%d_lr%1.0E_l2%1.0E' % (
            args.dataset, args.num, args.batch, args.lr, args.l2)
    else:
        save_dir = 'model/%s_num%d_batch%d_lr%1.0E_l2%1.0E' % (
            args.dataset, args.num, args.batch, args.lr, args.l2)

    #load data
    assert args.dataset in ['kuairand', 'coat', 'yahoo']
    if args.dataset == 'kuairand':
        data = loadData.load_kuairand(
            key=args.key, binary=args.binary, simple=False)
        # binary -> need key
        if args.binary:
            save_dir = 'model/%s_%s_num%d_batch%d_lr%1.0E_L2%1.0E' % (
                args.dataset, args.key, args.num, args.batch, args.lr, args.l2)
    elif args.dataset == 'coat':
        data = loadData.load_coat(binary=args.binary)
    else:
        data = loadData.load_yahoo(binary=args.binary)

    #dataset information
    num_user = int(np.amax(data[:, 0])) + 1
    num_item = int(np.amax(data[:, 1])) + 1
    data_shape = (num_user, num_item)
    print("User num:", num_user, "Item num:",
          num_item, "Dataset size:", len(data))

    if args.export:
        # do not split, just train and export rating matrix
        dataset = utils.MFDataset(data)
        data_loader = DataLoader(
            dataset, batch_size=args.batch, shuffle=True, num_workers=4)

        model = MFModel(num_user, num_item, args.num, np.mean(data[:, 2]))
        model.initial_args()
        criterion = torch.nn.MSELoss()
        optim = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.l2)

        for epoch in range(args.iter):
            model.train()
            loss = 0
            enumer = enumerate(data_loader)
            if args.verbose:
                enumer = tqdm(enumerate(data_loader))
            for Bid, batch in enumer:
                if args.verbose:
                    enumer.set_description("Batch id %d" % Bid)
                users, items, ratings = batch[0], batch[1], batch[2]
                #forward
                preds = model(users, items)
                l = criterion(preds, ratings)
                #backward
                optim.zero_grad()
                l.backward()
                optim.step()
                #show loss
                l = l.item()
                loss += l*len(users)
                if args.verbose:
                    enumer.set_postfix_str("l = %.4f" % l)
            #rmse
            train_rmse = (loss/len(data_loader.dataset))**0.5
            print("Train Iter %d, train rmse %.6f" % (epoch+1, train_rmse))

        #save matrix
        model.eval()
        users = torch.arange(num_user)
        items = torch.arange(num_item)
        pred_mx = np.zeros((num_user, num_item), dtype='float')
        known_mx = ss.csr_matrix(
            (data[:, 2], (data[:, 0].astype('int'), data[:, 1].astype('int')))).toarray()
        mask_mx = ss.csr_matrix((np.ones(len(data)), (data[:, 0].astype('int'), data[:, 1].astype('int')))).toarray()
        for start in range(0, num_user, 1024):
            end = start+1024 if start+1024 < num_user else num_user
            u_idx = torch.arange(start, end)
            data_mx = known_mx[u_idx]
            part_pred_mx = model.get_pred_mx(u_idx,mask_mx[u_idx],data_mx)
            pred_mx[u_idx] = part_pred_mx

        user_idxs = np.arange(num_user*num_item)
        user_idxs = np.floor_divide(user_idxs, num_item).astype('int32')
        item_idxs = np.arange(num_user*num_item)
        item_idxs = np.mod(item_idxs, num_item)
        pred_values = pred_mx.flatten()
        export = pd.DataFrame()
        export['user_id'] = user_idxs
        export['item_id'] = item_idxs
        export['value'] = pred_values
        if not os.path.exists('data'):
            os.mkdir('data')

        if args.dataset == 'kuairand' and args.binary:
            export_name = f'{args.dataset}_{args.key}'
        else:
            export_name = f'{args.dataset}' if args.binary else f'raw_{args.dataset}'
        export.to_csv('data/'+export_name+'.csv', index=False)
        print('ok!')
        exit()

    print('Optim!')
    #split data
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    valid, test = train_test_split(test, test_size=0.5, random_state=42)

    #make dataset
    train_ds = utils.MFDataset(train)
    valid_ds = utils.MFDataset(valid)
    test_ds = utils.MFDataset(test)

    #create dataloader
    test_batch = args.batch
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=test_batch)
    test_loader = DataLoader(test_ds, batch_size=test_batch)

    if not args.evaluate:
        #model
        model = MFModel(num_user, num_item, args.num, np.mean(train[:, 2]))
        model.initial_args()
        criterion = torch.nn.MSELoss()
        optim = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.l2)

        #train
        writer = SummaryWriter('runs/'+save_dir[6:]+str(time()))
        old_rmse = np.inf
        for epoch in range(args.iter):
            model.train()

            loss = 0
            enumer = enumerate(train_loader)
            if args.verbose:
                enumer = tqdm(enumerate(train_loader))
            for Bid, batch in enumer:
                if args.verbose:
                    enumer.set_description("Batch id %d" % Bid)
                users, items, ratings = batch[0], batch[1], batch[2]
                #forward
                preds = model(users, items)
                l = criterion(preds, ratings)
                #backward
                optim.zero_grad()
                l.backward()
                optim.step()
                #show loss
                l = l.item()
                loss += l*len(users)
                if args.verbose:
                    enumer.set_postfix_str("l = %.4f" % l)

            #rmse
            valid_rmse = utils.test_rating(model, valid_loader)
            train_rmse = (loss/len(train_loader.dataset))**0.5
            print("Train Iter %d, train rmse %.6f, valid rmse:%.6f" %
                  (epoch+1, train_rmse, valid_rmse))

            #write to tensorboard
            writer.add_scalar('RMSE/train', train_rmse, epoch)
            writer.add_scalar('RMSE/valid', valid_rmse, epoch)

            #early stop
            if valid_rmse > old_rmse:
                print("Early stop...")
                break

            old_rmse = valid_rmse

        writer.close()
        #save model
        if not os.path.exists('model'):
            os.mkdir('model')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(model.state_dict(), os.path.join(
            save_dir, 'MFmodel-ep%d.ckp' % args.iter))

    else:
        model = MFModel(num_user, num_item, args.num, np.mean(train[:, 2]))
        model.load_state_dict(torch.load(os.path.join(
            save_dir, 'MFmodel-ep%d.ckp' % args.iter)))
    #test & random
    test_rmse = utils.test_rating(model, test_loader)
    #random_rmse = utils.random_rating(test_loader,mean=np.mean(train[:, 2]))
    random_rmse = utils.random_rating(test_loader)
    print('random rmse:', random_rmse, 'test rmse:', test_rmse)

    """
    #evaluation
    #NDCG
    model.eval()
    print('NDCG @ \t2\t5\t10')
    print('MF\t%.2f\t%.2f\t%.2f' % (utils.NDCG_atK(model, data, 2),
                                    utils.NDCG_atK(model, data, 5),
                                    utils.NDCG_atK(model, data, 10)))
    print('Random\t%.2f\t%.2f\t%.2f' % (utils.random_NDCG_atK(model, data, 2),
                                        utils.random_NDCG_atK(model, data, 5),
                                        utils.random_NDCG_atK(model, data, 10)))

    # AUC HR Recall Prec
    """

    # group visual
    if args.dataset == 'coat':
        group_by = [0, 20, 30, 40, 50, 60]
    if args.dataset == 'yahoo':
        group_by = [0, 100, 200, 300, 400, 500, 600, 700, 800]
        group_by = [0, 100, 200, 300, 400, 500, 800]
    if args.dataset == 'kuairand':
        group_by = [0, 100, 200, 300, 400, 500, 600]
    if args.dataset == 'kuairand' and args.binary:
        visaul_name = f'{args.dataset}_{args.key}_visual'
    else:
        visaul_name = f'{args.dataset}_visual' if args.binary else f'raw_{args.dataset}_visual'
    utils.visual(train, (num_user, num_item), group_by,
                 model, save_name=visaul_name)
