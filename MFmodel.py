import torch


class MFModel(torch.nn.Module):

    def __init__(self, n_user, n_item, n_latent, mean):
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_latent = n_latent
        self.mean = mean

        # 2 matrixs
        self.u_mx = torch.nn.Embedding(self.n_user, self.n_latent)
        self.v_mx = torch.nn.Embedding(self.n_item, self.n_latent)

        # user bias & item bias
        self.u_bias = torch.nn.Embedding(self.n_user, 1)
        self.v_bias = torch.nn.Embedding(self.n_item, 1)

    def initial_args(self, init_std=0.01):
        self.u_mx.weight.data = torch.normal(
            0, init_std, size=(self.n_user, self.n_latent))
        self.v_mx.weight.data = torch.normal(
            0, init_std, size=(self.n_item, self.n_latent))
        self.u_bias.weight.data = torch.zeros(self.n_user, 1).float()
        self.v_bias.weight.data = torch.zeros(self.n_item, 1).float()

    def forward(self, u_idx, v_idx):
        u_vec = self.u_mx(u_idx)
        v_vec = self.v_mx(v_idx)
        pred = torch.mul(u_vec, v_vec).sum(dim=1) + self.mean + \
            self.u_bias(u_idx).reshape(-1) + self.v_bias(v_idx).reshape(-1)

        return pred

    def topk(self, u_idx, k, data_mx = None):
        # u_idx: torch.tensor(n)
        # k: item num
        # data_mx: train_data matrix (nxV), 1 for exits in dataset, 0 for not
        # return top-k ratings, top-k indexs
        # both shape (len(u_idx),k)
        # Note that items in train data will not be recommended again
        user_mx = self.u_mx(u_idx)
        item_mx = self.v_mx(torch.arange(self.n_item))
        pred_mx = torch.matmul(user_mx, item_mx.T)
        user_bias = torch.mul(self.u_bias(u_idx), torch.ones_like(pred_mx))
        item_bias = torch.mul(self.v_bias(torch.arange(
            self.n_item)).reshape(-1), torch.ones_like(pred_mx))
        pred_mx += user_bias + item_bias + self.mean
        if data_mx is not None:
            # set pred of (u,v) in train_data with value 0 (thus not picked)
            pred_mx = torch.mul(pred_mx,torch.tensor(data_mx==0))
        return torch.topk(pred_mx, k)
