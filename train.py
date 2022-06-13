import time
import torch
import numpy as np
from dataset import *
from torch.utils.data import DataLoader
from model.trainnet import *
from tqdm import tqdm
import random
from copy import deepcopy

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
    trainset = CUB_200_2011(train=True, index_path="data/CUB_200_2011/index_list/cub200/session_1.txt")
    testset = CUB_200_2011(train=False, index=np.arange(base_class))

    sampler = CategoriesSampler(trainset.targets, episode_num, episode_way, episode_shot + episode_query)

    trainloader = DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=num_workers, pin_memory=True)

    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader(session, batch_size, num_workers, base_class, way):
    trainset = CUB_200_2011(train=True, index_path="data/CUB_200_2011/index_list/cub200/session_" + str(session + 1) + ".txt")
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
    optimizer = torch.optim.SGD([{'params': model.module.backbone.parameters(), 'lr': 0.0002}, {'params': model.module.slf_attn.parameters(), 'lr': 0.0002}], momentum=0.9, nesterov=True, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    return optimizer, scheduler

def replace_to_rotate(proto_tmp, query_tmp, low_way):
    for i in range(low_way):
        # random choose rotate degree
        rot_list = [90, 180, 270]
        sel_rot = random.choice(rot_list)
        if sel_rot == 90:  # rotate 90 degree
            # print('rotate 90 degree')
            proto_tmp[i::low_way] = proto_tmp[i::low_way].transpose(2, 3).flip(2)
            query_tmp[i::low_way] = query_tmp[i::low_way].transpose(2, 3).flip(2)
        elif sel_rot == 180:  # rotate 180 degree
            # print('rotate 180 degree')
            proto_tmp[i::low_way] = proto_tmp[i::low_way].flip(2).flip(3)
            query_tmp[i::low_way] = query_tmp[i::low_way].flip(2).flip(3)
        elif sel_rot == 270:  # rotate 270 degree
            # print('rotate 270 degree')
            proto_tmp[i::low_way] = proto_tmp[i::low_way].transpose(2, 3).flip(3)
            query_tmp[i::low_way] = query_tmp[i::low_way].transpose(2, 3).flip(3)
    return proto_tmp, query_tmp

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def base_train(model, trainloader, optimizer, scheduler, epoch, episode_way, episode_shot, episode_query, low_way, low_shot):
    training_loss = Averager()
    training_acc = Averager()

    tqdm_gen = tqdm(trainloader)

    label = torch.arange(episode_way + low_way).repeat(episode_query)
    label = label.type(torch.cuda.LongTensor)

    for i, batch in enumerate(tqdm_gen, 1):
        data, true_label = [_.cuda() for _ in batch]
        del batch
        torch.cuda.empty_cache()

        k = episode_way * episode_shot
        proto, query = data[:k], data[k:]
        # sample low_way data
        proto_tmp = deepcopy(
            proto.reshape(episode_shot, episode_way, proto.shape[1], proto.shape[2], proto.shape[3])[
            :low_shot,
            :low_way, :, :, :].flatten(0, 1))
        query_tmp = deepcopy(
            query.reshape(episode_query, episode_way, query.shape[1], query.shape[2], query.shape[3])[:,
            :low_way, :, :, :].flatten(0, 1))
        
        # random choose rotate degree
        proto_tmp, query_tmp = replace_to_rotate(proto_tmp, query_tmp, low_way)
        model.module.mode = 'encoder'

        data = model(data)
        proto_tmp = model(proto_tmp)
        query_tmp = model(query_tmp)

        # k = episode_way * episode_shot
        proto, query = data[:k], data[k:]

        proto = proto.view(episode_shot, episode_way, proto.shape[-1])
        query = query.view(episode_query, episode_way, query.shape[-1])

        proto_tmp = proto_tmp.view(low_shot, low_way, proto.shape[-1])
        query_tmp = query_tmp.view(episode_query, low_way, query.shape[-1])

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
        tqdm_gen.set_description('Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        training_loss.add(total_loss.item())
        training_acc.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    training_loss = training_loss.item()
    training_acc = training_acc.item()
    return training_loss, training_acc

def replace_base_fc(trainset, transform, model, base_class, batch_size, num_workers):
    model = model.eval()

    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:base_class] = proto_list

    return model

def test(model, testloader, base_class, way, session):
    test_class = base_class + session * way
    model = model.eval()
    validation_loss = Averager()
    validation_acc = Averager()
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]

            model.module.mode = 'encoder'
            query = model(data)
            query = query.unsqueeze(0).unsqueeze(0)

            proto = model.module.fc.weight[:test_class, :].detach()
            proto = proto.unsqueeze(0).unsqueeze(0)

            logits = model.module._forward(proto, query)

            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            validation_loss.add(loss.item())
            validation_acc.add(acc)

        validation_loss = validation_loss.item()
        validation_acc = validation_acc.item()

    return validation_loss, validation_acc

def validation(model, episode_num, episode_way, episode_shot, episode_query, batch_size, num_workers, base_class, way, sessions):
    with torch.no_grad():

        for session in range(1, sessions):
            trainset, trainloader, testloader = get_dataloader(session, episode_num, episode_way, episode_shot, episode_query, batch_size, num_workers, base_class, way)

            trainloader.dataset.transform = testloader.dataset.transform
            model.module.mode = 'avg_cos'
            model.eval()
            model.module.update_fc(trainloader, np.unique(trainset.targets))

            validation_loss, validation_acc = test(model, testloader, base_class, way, session)

    return validation_loss, validation_acc

if __name__ == '__main__':
    ## start time
    t_start_time = time.time()

    ## num_gpu
    num_gpu = torch.cuda.device_count()

    ## episode training parameters
    episode_num = 50
    episode_way = 10
    episode_shot = 1
    episode_query = 10

    ## low
    low_way = 10
    low_shot = 1

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
    num_workers = max(round(os.cpu_count() / 2), 2)

    save_path = './trained_model'

    model = TRAINNET(mode='ft_cos')
    model = nn.DataParallel(model, list(range(num_gpu)))
    model = model.cuda()

    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['test_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['test_acc'] = []
    trlog['max_acc_epoch'] = 0
    trlog['max_acc'] = [0.0] * session_number

    model_dir = './pretrained_model/session0_max_acc.pth'

    best_model_dict = torch.load(model_dir)['params']

    for session in range(start_session, session_number):
        trainset, trainloader, testloader = get_dataloader(session, episode_num, episode_way, episode_shot, episode_query, batch_size, num_workers, base_class, way)
        model = update_param(model, best_model_dict)

        if session == 0:
            print('new classes for this session:\n', np.unique(trainset.targets))
            optimizer, scheduler = get_optimizer_base(model, milestones)

            for epoch in range(train_epochs):
                start_time = time.time()

                model.eval()
                training_loss, training_acc = base_train(model, trainloader, optimizer, scheduler, epoch, episode_way, episode_shot, episode_query, low_way, low_shot)
                model = replace_base_fc(trainset, testloader.dataset.transform, model, base_class, batch_size, num_workers)
                model.module.mode = 'avg_cos'

                validation_loss, validation_acc = validation(model, episode_num, episode_way, episode_shot, episode_query, batch_size, num_workers, base_class, way, session_number)

                # save better model
                if (validation_acc * 100) >= trlog['max_acc'][session]:
                    trlog['max_acc'][session] = float('%.3f' % (validation_acc * 100))
                    trlog['max_acc_epoch'] = epoch
                    save_model_dir = os.path.join(save_path, 'session' + str(session) + '_max_acc.pth')
                    torch.save(dict(params = model.state_dict()), save_model_dir)
                    torch.save(optimizer.state_dict(), os.path.join(save_path, 'optimizer_best.pth'))
                    best_model_dict = deepcopy(model.state_dict())
                    print('********A better model is found!!**********')
                    print('Saving model to :%s' % save_model_dir)
                print('best epoch {}, best val acc={:.3f}'.format(trlog['max_acc_epoch'], trlog['max_acc'][session]))
                trlog['val_loss'].append(validation_loss)
                trlog['val_acc'].append(validation_acc)
                lrc = scheduler.get_last_lr()[0]
                print('epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (epoch, lrc, training_loss, training_acc, validation_loss, validation_acc))

                trlog['train_loss'].append(training_loss)
                trlog['train_acc'].append(training_acc)

                print('This epoch takes %d seconds' % (time.time() - start_time), '\nstill need around %.2f mins to finish' % ((time.time() - start_time) * (train_epochs - epoch) / 60))
                scheduler.step()

                # always replace fc with avg mean
                model.load_state_dict(best_model_dict)
                model = replace_base_fc(trainset, testloader.dataset.transform, model, base_class, batch_size, num_workers)
                best_model_dir = os.path.join(save_path, 'session' + str(session) + '_max_acc.pth')
                print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                best_model_dict = deepcopy(model.state_dict())
                torch.save(dict(params=model.state_dict()), best_model_dir)

                model.module.mode = 'avg_cos'
                test_loss, test_acc = test(model, testloader, base_class, way, session)
                trlog['max_acc'][session] = float('%.3f' % (test_acc * 100))
                print('The test acc of base session={:.3f}'.format(trlog['max_acc'][session]))

        else:  # incremental learning sessions
            print("training session: [%d]" % session)
            model.load_state_dict(best_model_dict)

            model.module.mode = 'avg_cos'
            model.eval()
            trainloader.dataset.transform = testloader.dataset.transform
            model.module.update_fc(trainloader, np.unique(trainset.targets))

            test_loss, test_acc = test(model, testloader, base_class, way, session)

            # save better model
            trlog['max_acc'][session] = float('%.3f' % (test_acc * 100))

            save_model_dir = os.path.join(save_path, 'session' + str(session) + '_max_acc.pth')
            torch.save(dict(params=model.state_dict()), save_model_dir)
            best_model_dict = deepcopy(model.state_dict())
            print('Saving model to :%s' % save_model_dir)
            print('  test acc={:.3f}'.format(trlog['max_acc'][session]))

    print(trlog['max_acc'])

    t_end_time = time.time()
    total_time = (t_end_time - t_start_time) / 60
    print('Best epoch:', trlog['max_acc_epoch'])
    print('Total time used %.2f mins' % total_time)