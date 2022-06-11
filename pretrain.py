import numpy as np
from dataset import *
from torch.utils.data import DataLoader
from model.pretrainnet import *
from copy import deepcopy
import time
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

## base class dataloader
def get_base_dataloader():
    trainset = CUB_200_2011(train=True, index=np.arange(100), base_sess=True)
    testset = CUB_200_2011(train=False, index=np.arange(100))

    trainloader = DataLoader(dataset=trainset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    testloader = DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader

## novel class dataloader
def get_novel_dataloader(session):
    trainset = CUB_200_2011(train=True, index_path="data/index_list/cub200/session_" + str(session + 1) + '.txt')

    trainloader = DataLoader(dataset=trainset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

    class_new = np.arange(100 + session * 10)

    testset = CUB_200_2011(train=False, index=class_new)

    testloader = DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader

## dataloader for each session
def get_dataloader(session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader()
    else:
        trainset, trainloader, testloader = get_novel_dataloader(session)
    return trainset, trainloader, testloader

## optimzier, scheduler initialize
def get_optimizer_base(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40, 60, 80], gamma=0.1)

    return optimizer, scheduler

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def base_train(model, trainloader, optimizer, scheduler, epoch):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        logits = model(data)
        logits = logits[:, :100]
        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)

        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta

def test(model, testloader, epoch, way, session):
    test_class = 100 + session * way
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va

def replace_base_fc(trainset, transform, model):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, num_workers=8, pin_memory=True, shuffle=False)
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

    for class_index in range(100):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:100] = proto_list

    return model

if __name__ == '__main__':
    
    t_start_time = time.time()

    ## number of gpu
    num_gpu = 1

    ## number of epoch
    train_epochs = 100

    ## way
    way = 10

    ## shot
    shot = 5

    save_path = './model_saved'
    
    ## model initialize
    model = PRETRAINNET(mode='ft_cos')
    model = nn.DataParallel(model, list(range(num_gpu)))
    model = model.cuda()

    ## best model
    best_model_dict = deepcopy(model.state_dict())
    
    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['test_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['test_acc'] = []
    trlog['max_acc_epoch'] = 0
    trlog['max_acc'] = [0.0] * 10

    for session in range(0, 11):
        ## load dataset
        trainset, trainloader, testloader = get_dataloader(session)
        
        ## update model state
        model.load_state_dict(best_model_dict)

        if session == 0:
            optimizer, scheduler = get_optimizer_base(model)

            for epoch in range(train_epochs):
                start_time = time.time()
                
                # train base sess
                tl, ta = base_train(model, trainloader, optimizer, scheduler, epoch)
                # test model with all seen class
                tsl, tsa = test(model, testloader, epoch, way, session)

                # save better model
                if (tsa * 100) >= trlog['max_acc'][session]:
                    trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                    trlog['max_acc_epoch'] = epoch
                    save_model_dir = os.path.join(save_path, 'session' + str(session) + '_max_acc.pth')
                    torch.save(dict(params=model.state_dict()), save_model_dir)
                    torch.save(optimizer.state_dict(), os.path.join(save_path, 'optimizer_best.pth'))
                    best_model_dict = deepcopy(model.state_dict())
                    print('********A better model is found!!**********')
                    print('Saving model to :%s' % save_model_dir)
                print('best epoch {}, best test acc={:.3f}'.format(trlog['max_acc_epoch'], trlog['max_acc'][session]))

                trlog['train_loss'].append(tl)
                trlog['train_acc'].append(ta)
                trlog['test_loss'].append(tsl)
                trlog['test_acc'].append(tsa)
                lrc = scheduler.get_last_lr()[0]
                print('This epoch takes %d seconds' % (time.time() - start_time), '\nstill need around %.2f mins to finish this session' % ((time.time() - start_time) * (train_epochs - epoch) / 60))
                scheduler.step()

            model.load_state_dict(best_model_dict)
            model = replace_base_fc(trainset, testloader.dataset.transform, model)
            best_model_dir = os.path.join(save_path, 'session' + str(session) + '_max_acc.pth')
            print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
            best_model_dict = deepcopy(model.state_dict())
            torch.save(dict(params=model.state_dict()), best_model_dir)

            model.module.mode = 'avg_cos'
            tsl, tsa = test(model, testloader, 0, way, session)
            if (tsa * 100) >= trlog['max_acc'][session]:
                trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                print('The new best test acc of base session={:.3f}'.format(trlog['max_acc'][session]))


        else:  # incremental learning sessions
            print("training session: [%d]" % session)

            model.module.mode = 'avg_cos'
            model.eval()

            trainloader.dataset.transform = testloader.dataset.transform
            model.module.update_fc(trainloader, np.unique(trainset.targets), session)

            tsl, tsa = test(model, testloader, 0, way, session)

            # save model
            trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
            save_model_dir = os.path.join(save_path, 'session' + str(session) + '_max_acc.pth')
            torch.save(dict(params=model.state_dict()), save_model_dir)
            best_model_dict = deepcopy(model.state_dict())
            print('Saving model to :%s' % save_model_dir)
            print('  test acc={:.3f}'.format(trlog['max_acc'][session]))

    t_end_time = time.time()
    total_time = (t_end_time - t_start_time) / 60
    print('Base Session Best epoch:', trlog['max_acc_epoch'])
    print('Total time used %.2f mins' % total_time)