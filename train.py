import time
import torch
import numpy as np
from dataset import *
from model.trainnet import *

class CategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indexs,e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

def get_base_dataloader_meta(episode_num, episode_way, episode_shot, episode_query):
    trainset = CUB_200_2011(train=True, index_path="data/index_list/cub200/session_1.txt")
    testset = CUB_200_2011(train=False, index=np.arange(100))

    sampler = CategoriesSampler(trainset.targets, episode_num, episode_way, episode_shot + episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=2, pin_memory=True)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return trainset, trainloader, testloader

def get_dataloader(session, episode_num, episode_way, episode_shot, episode_query):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader_meta(episode_num, episode_way, episode_shot, episode_query)
    else:
        trainset, trainloader, testloader = get_new_dataloader(session)
    return trainset, trainloader, testloader

if __name__ == '__main__':
    ## start time
    t_start_time = time.time()

    ## num_gpu
    num_gpu = 1

    episode_num = 50
    episode_way = 10
    episode_shot = 5
    episode_query = 10

    model = TRAINNET(mode='ft_cos')
    model = nn.DataParallel(model, list(range(num_gpu)))
    model = model.cuda()

    model_dir = './model_saved/session0_max_acc.pth'

    best_model_dict = torch.load(model_dir)['params']

    for session in range(11):
        trainset, trainloader, testlaoder = get_dataloader(session, episode_num, episode_way, episode_shot, episode_query)
        model = update_param(model, best_model_dict)