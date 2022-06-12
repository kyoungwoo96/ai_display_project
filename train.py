import time
import torch
import numpy as np
from dataset import *
from torch.utils.data import DataLoader
from model.trainnet import *
from tqdm import tqdm

class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

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
        for _ in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indexs,e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

def get_base_dataloader_meta(base_class, episode_num, episode_way, episode_shot, episode_query, batch_size, num_workers):
    trainset = CUB_200_2011(train=True, index_path="data/index_list/cub200/session_1.txt")
    testset = CUB_200_2011(train=False, index=np.arange(base_class))

    sampler = CategoriesSampler(trainset.targets, episode_num, episode_way, episode_shot + episode_query)

    trainloader = DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=num_workers, pin_memory=True)

    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader(session, batch_size, num_workers, base_class, way):
    trainset = CUB_200_2011(train=True, index_path="data/index_list/cub200/session_" + str(session + 1) + ".txt")
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = CUB_200_2011(train=False, index=np.arange(base_class + session * way))
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_dataloader(session, episode_num, episode_way, episode_shot, episode_query, batch_size, num_workers, base_class, way):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader_meta(base_class, episode_num, episode_way, episode_shot, episode_query, batch_size, num_workers)
    else:
        trainset, trainloader, testloader = get_new_dataloader(session, batch_size, num_workers, base_class, way)
    return trainset, trainloader, testloader

def update_param(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def get_optimizer_base(model, milestones):
    optimizer = torch.optim.SGD([{'params': model.module.encoder.parameters(), 'lr': 0.0002}, {'params': model.module.slf_attn.parameters(), 'lr': 0.0002}], momentum=0.9, nesterov=True, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    return optimizer, scheduler

def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()

    tqdm_gen = tqdm(trainloader)

    label = torch.arange(args.episode_way + args.low_way).repeat(args.episode_query)
    label = label.type(torch.cuda.LongTensor)

    for i, batch in enumerate(tqdm_gen, 1):
        data, true_label = [_.cuda() for _ in batch]

        k = args.episode_way * args.episode_shot
        proto, query = data[:k], data[k:]
        # sample low_way data
        proto_tmp = deepcopy(
            proto.reshape(args.episode_shot, args.episode_way, proto.shape[1], proto.shape[2], proto.shape[3])[
            :args.low_shot,
            :args.low_way, :, :, :].flatten(0, 1))
        query_tmp = deepcopy(
            query.reshape(args.episode_query, args.episode_way, query.shape[1], query.shape[2], query.shape[3])[:,
            :args.low_way, :, :, :].flatten(0, 1))
        # random choose rotate degree
        proto_tmp, query_tmp = self.replace_to_rotate(proto_tmp, query_tmp)
        model.module.mode = 'encoder'

        data = model(data)
        proto_tmp = model(proto_tmp)
        query_tmp = model(query_tmp)

        # k = args.episode_way * args.episode_shot
        proto, query = data[:k], data[k:]

        proto = proto.view(args.episode_shot, args.episode_way, proto.shape[-1])
        query = query.view(args.episode_query, args.episode_way, query.shape[-1])

        proto_tmp = proto_tmp.view(args.low_shot, args.low_way, proto.shape[-1])
        query_tmp = query_tmp.view(args.episode_query, args.low_way, query.shape[-1])

        proto = proto.mean(0).unsqueeze(0)
        proto_tmp = proto_tmp.mean(0).unsqueeze(0)

        proto = torch.cat([proto, proto_tmp], dim=1)
        query = torch.cat([query, query_tmp], dim=1)

        proto = proto.unsqueeze(0)
        query = query.unsqueeze(0)

        logits = model.module._forward(proto, query)

        total_loss = F.cross_entropy(logits, label)

        acc = count_acc(logits, label)

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta

if __name__ == '__main__':
    ## start time
    t_start_time = time.time()

    ## num_gpu
    num_gpu = 1

    ## episode training parameters
    episode_num = 50
    episode_way = 10
    episode_shot = 5
    episode_query = 10

    ## number of epoch
    train_epochs = 100

    ## start_session
    start_session = 0

    ## session_number
    session_number = 11

    ## base_class
    base_class = 100

    ## milestones
    milestones = [30, 40, 60, 80]

    ## incremental learning way
    way = 10

    ## batch_size
    batch_size = 64

    ## number of dataset load workers
    num_workers = 2

    model = TRAINNET(mode='ft_cos')
    model = nn.DataParallel(model, list(range(num_gpu)))
    model = model.cuda()

    model_dir = './model_saved/session0_max_acc.pth'

    best_model_dict = torch.load(model_dir)['params']

    for session in range(start_session, session_number):
        trainset, trainloader, testlaoder = get_dataloader(session, episode_num, episode_way, episode_shot, episode_query, batch_size, num_workers, base_class, way)
        model = update_param(model, best_model_dict)

        if session == 0:
            print('new classes for this session:\n', np.unique(trainset.targets))
            optimizer, scheduler = get_optimizer_base(model, milestones)

            for epoch in range(train_epochs):
                start_time = time.time()

                model.eval()
                training_loss, training_acc = base_train(model, trainloader, optimizer, scheduler, epoch, args)