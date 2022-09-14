import torch

class MFDataset():

    def __init__(self,data):
        # data: array of (u,v,rating)
        self.users = torch.LongTensor(data[:,0])
        self.items = torch.LongTensor(data[:,1])
        self.ratings = torch.FloatTensor(data[:,2])

    def __getitem__(self,index):
        return self.users[index],self.items[index],self.ratings[index]

    def __len__(self):
        return len(self.users)

def test_rating(model,data_loader):
    # model : model used to predict, model(uidx,vidx)
    # data_loader : torch.dataloader
    # return : rmse
    model.eval()
    with torch.no_grad():
        loss = 0
        for batch in data_loader:
            users,items,ratings = batch[0],batch[1],batch[2]
            preds = model(users,items)
            loss += torch.sum((preds-ratings)**2).item()
            
    return (loss/len(data_loader.dataset))**0.5

