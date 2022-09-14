import os
import torch
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from MFmodel import MFModel
import loadData
import utils

if __name__ == "__main__":
    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="yahoo")  # dataset
    # only useful for KuaiRand
    parser.add_argument("--key", default="is_click")

    # dims of latent factor
    parser.add_argument("--num", default=50, type=int)
    parser.add_argument("--iter", default=10, type=int)  # max iteration
    parser.add_argument("--batch", default=1024, type=int)  # batch size
    parser.add_argument("--lr", default=1e-3, type=float)  # learning rate
    parser.add_argument("--l2",default=1e-5,type=float) # L2 regularization
    parser.add_argument("--verbose",default=False,type=bool) # args.verbose
    args = parser.parse_args()

    save_dir = 'model/%s_num%d_batch%d_lr%1.0E_l2%1.0E' % (
        args.dataset, args.num, args.batch, args.lr, args.l2)

    #load data
    assert args.dataset in ['kuairand', 'coat', 'yahoo']
    if args.dataset == 'kuairand':
        data = loadData.load_kuairand(key=args.key)
        save_dir = 'model/%s_%s_num%d_batch%d_lr%1.0E_L2%1.0E' % (
            args.dataset, args.key, args.num, args.batch, args.lr, args.l2)
    elif args.dataset == 'coat':
        data = loadData.load_coat()
    else:
        data = loadData.load_yahoo()

    #dataset information
    num_user = np.amax(data[:, 0]) + 1
    num_item = np.amax(data[:, 1]) + 1
    data_shape = (num_user, num_item)

    #split data
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    valid, test = train_test_split(test, test_size=0.5,random_state=42)

    #make dataset
    train_ds = utils.MFDataset(train)
    valid_ds = utils.MFDataset(valid)
    test_ds = utils.MFDataset(test)

    #create dataloader
    test_batch = args.batch
    train_loader = DataLoader(train_ds,batch_size=args.batch,shuffle=False,num_workers=0)
    valid_loader = DataLoader(valid_ds,batch_size=test_batch)
    test_loader = DataLoader(test_ds,batch_size=test_batch)

    #model
    model = MFModel(num_user,num_item,args.num,np.mean(train[:,2]))
    model.initial_args()
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    #train
    writer = SummaryWriter()
    for epoch in range(args.iter):
        model.train()

        loss = 0
        enumer = enumerate(train_loader)
        if args.verbose:
            enumer = tqdm(enumerate(train_loader))
        for Bid, batch in enumer:
            if args.verbose:
                enumer.set_description("Batch id %d"%Bid)
            users,items,ratings = batch[0],batch[1],batch[2]
            #forward
            preds = model(users,items)
            l = criterion(preds,ratings)
            #backward
            optim.zero_grad()
            l.backward()
            optim.step()
            #show loss
            l = l.item()
            loss += l*len(users)
            if args.verbose:
                enumer.set_postfix_str("l = %.4f"%l)

        #rmse
        valid_rmse = utils.test_rating(model,valid_loader)
        train_rmse = (loss/len(train_loader.dataset))**0.5
        print("Train Iter %d, train rmse %.6f, valid rmse:%.6f"%(epoch,train_rmse,valid_rmse))

        #write to tensorboard
        writer.add_scalar('RMSE/train',train_rmse,epoch)
        writer.add_scalar('RMSE/valid',valid_rmse,epoch)

    writer.close()
    #save model
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(model.state_dict(),os.path.join(save_dir,'MFmodel-ep%d.ckp'%args.iter))

    #test
    test_rmse = utils.test_rating(model,test_loader)
    #random
    pred = np.random.random((len(test),1))
    random_rmse = np.sqrt(np.mean((test[:,2]-pred)**2))
    print('random rmse:',random_rmse,'test rmse:',test_rmse)

    #evaluation
    #NDCG AUC HR Recall Prec



        