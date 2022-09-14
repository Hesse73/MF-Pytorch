import torch

class MFModel(torch.nn.Module):

    def __init__(self,n_user,n_item,n_latent,mean):
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_latent = n_latent
        self.mean = mean

        # 2 matrixs
        self.u_mx = torch.nn.Embedding(self.n_user,self.n_latent)
        self.v_mx = torch.nn.Embedding(self.n_item,self.n_latent)

        # user bias & item bias
        self.u_bias = torch.nn.Embedding(self.n_user,1)
        self.v_bias = torch.nn.Embedding(self.n_item,1)

    def initial_args(self,init_std=0.01):
        self.u_mx.weight.data = torch.normal(0,init_std,size=(self.n_user,self.n_latent))
        self.v_mx.weight.data = torch.normal(0,init_std,size=(self.n_item,self.n_latent))
        self.u_bias.weight.data = torch.zeros(self.n_user,1).float()
        self.v_bias.weight.data = torch.zeros(self.n_item,1).float()

    def forward(self,u_idx,v_idx):
        u_vec = self.u_mx(u_idx)
        v_vec = self.v_mx(v_idx)
        pred = torch.mul(u_vec,v_vec).sum(dim=1) + self.mean + self.u_bias(u_idx).reshape(-1) + self.v_bias(v_idx).reshape(-1)
        
        return pred



