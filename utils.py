import torch
import os
import scipy.sparse as ss
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


class MFDataset():

    def __init__(self, data):
        # data: array of (u,v,rating)
        self.users = torch.LongTensor(data[:, 0])
        self.items = torch.LongTensor(data[:, 1])
        self.ratings = torch.FloatTensor(data[:, 2])

    def __getitem__(self, index):
        return self.users[index], self.items[index], self.ratings[index]

    def __len__(self):
        return len(self.users)


def test_rating(model, data_loader):
    # model : model used to predict, model(uidx,vidx)
    # data_loader : torch.dataloader
    # return : rmse
    model.eval()
    with torch.no_grad():
        loss = 0
        for batch in data_loader:
            users, items, ratings = batch[0], batch[1], batch[2]
            preds = model(users, items)
            loss += torch.sum((preds-ratings)**2).item()

    return (loss/len(data_loader.dataset))**0.5


def random_rating(data_loader,mean=None):
    loss = 0
    for batch in data_loader:
        _, _, ratings = batch[0], batch[1], batch[2]
        if mean is not None:
            preds = torch.normal(mean,0.01,ratings.shape)
            preds[preds>1] = 1
            preds[preds<0] = 0
        else:
            preds = torch.rand_like(ratings)
        loss += torch.sum((preds-ratings)**2).item()
    return (loss/len(data_loader.dataset))**0.5


def NDCG_atK(model, data, k=5, batch_size=1024):
    model.eval()

    n_user, n_item = model.n_user, model.n_item
    result = 0

    data_mx = ss.csr_matrix((data[:, 2], (data[:, 0], data[:, 1])))
    for start in range(0, n_user, batch_size):
        end = start+batch_size if start+batch_size < n_user else n_user-1
        u_idx = torch.arange(start, end)
        _, indexs = model.topk(u_idx, k)
        #get real values
        cur_mx = data_mx[u_idx].toarray()
        real_vals = cur_mx[torch.arange(0, end-start).reshape(-1, 1), indexs]
        #calculate DCG@k
        mask = torch.mul(torch.arange(1, k+1), torch.ones_like(indexs))
        mask = torch.log2(mask+1)
        dcg = (real_vals/mask).sum(dim=1)
        #calculate iDCG@k
        cur_mx[:, ::-1].sort()
        idcg = (cur_mx[:, :k]/mask).sum(dim=1)
        #nDCG@k for this batch
        result += torch.nansum(dcg/idcg).item()

    return result/n_user


def random_NDCG_atK(model, data, k=5, batch_size=1024):

    n_user, n_item = model.n_user, model.n_item
    result = 0

    data_mx = ss.csr_matrix((data[:, 2], (data[:, 0], data[:, 1])))
    for start in range(0, n_user, batch_size):
        end = start+batch_size if start+batch_size < n_user else n_user-1
        u_idx = torch.arange(start, end)
        indexs = torch.randint(n_item, (len(u_idx), k))
        #get real values
        cur_mx = data_mx[u_idx].toarray()
        real_vals = cur_mx[torch.arange(0, end-start).reshape(-1, 1), indexs]
        #calculate DCG@k
        mask = torch.mul(torch.arange(1, k+1), torch.ones_like(indexs))
        mask = torch.log2(mask+1)
        dcg = (real_vals/mask).sum(dim=1)
        #calculate iDCG@k
        cur_mx[:, ::-1].sort()
        idcg = (cur_mx[:, :k]/mask).sum(dim=1)
        #nDCG@k for this batch
        result += torch.nansum(dcg/idcg).item()

    return result/n_user

def visual(train_data,dataset_shape,group_by,model,save_name='visual'):

    num_user,num_item = dataset_shape
    test_batch = 1024
    # Analyse exposure - recommendation counts
    # 1. get exposure
    expo_items,exposures = np.unique(train_data[:,1].astype('int'),return_counts=True)
    temp = np.zeros(num_item,dtype='int')
    temp[expo_items] = exposures
    exposures = temp
    assert len(exposures) == num_item
    # 2. recommendate for every user, get item counts
    counts = np.zeros(num_item)
    contrast_counts = np.zeros(num_item)
    train_mx= ss.csr_matrix((train_data[:,2],(train_data[:,0].astype('int'),train_data[:,1].astype('int'))))
    train_mx.resize((num_user,num_item))
    for start in range(0, num_user, test_batch):
        end = start+test_batch if start+test_batch < num_user else num_user-1
        u_idx = torch.arange(start, end)
        data_mx = train_mx[u_idx].toarray()
        _,pred_items = model.topk(u_idx,5,data_mx)
        pred_items,pred_counts = torch.unique(pred_items.reshape(-1),return_counts=True)
        pred_items,pred_counts = pred_items.numpy(),pred_counts.numpy()
        counts[pred_items] += pred_counts
        # random for contrast
        rand_items = np.random.randint(0,num_item,(len(u_idx),5))
        rand_items,rand_counts = np.unique(rand_items.reshape(-1),return_counts=True)
        contrast_counts[rand_items] += rand_counts
    assert np.sum(counts) == np.sum(contrast_counts)
    # 3. group by exposure
    #group
    group_by = group_by.copy()
    group_by.append(np.max(exposures)+1)
    group_info = []
    group_item_num = []
    group_rec_count = []
    group_contrast_count = []
    for start,end in zip(group_by[:-1],group_by[1:]):
        group_info.append(f"[{start},{end})")
        group_idxs = np.where((exposures>=start)&(exposures<end))[0]
        g_item_num = len(group_idxs)
        if(g_item_num<10):
            print('Warning: group[%d,%d) only have %d items!'%(start,end,g_item_num))
        if(g_item_num==0):
            group_item_num.append(g_item_num)
            group_rec_count.append(np.nan)
            group_contrast_count.append(np.nan)
            continue
        g_counts = counts[group_idxs]
        group_item_num.append(g_item_num)
        group_rec_count.append(np.sum(g_counts)/g_item_num)
        group_contrast_count.append(np.sum(contrast_counts[group_idxs])/g_item_num)
    #4. show fig
    sns.set_theme(style="dark")
    group_df = pd.DataFrame([group_info,group_item_num,group_rec_count,group_contrast_count],index=['group','item_num','rec_count','contrast_count']).T
    fig = plt.figure()
    sns.barplot(data=group_df,x='group',y='item_num')
    ax1=plt.gca()
    ax2 = ax1.twinx()
    sns.lineplot(data=group_df,x='group',y='rec_count',ax=ax2,label='MF rec')
    sns.lineplot(data=group_df,x='group',y='contrast_count',ax=ax2,label='Random')
    plt.legend()
    plt.show()
    if not os.path.exists('pics'):
        os.mkdir('pics')
    fig.savefig(f"pics/{save_name}.pdf", format='pdf', bbox_inches='tight')
    fig.savefig(f"pics/{save_name}.png", format='png', bbox_inches='tight')