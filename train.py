import torch
import os
import torch.nn as nn
from torchvision.models import resnet34
from torchvision.transforms import transforms

import utils
from config import config
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader,SubsetRandomSampler

from valid import valid
from hovertrans import hovernet
from utils import confusion_matrix
import math


def train(config, train_loader, test_loader, fold, test_idx):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # NET
    # net = hovernet(img_size=config.img_size, num_classes=config.class_num, drop_rate=0.1, attn_drop_rate=0.1,
    #         patch_size=config.patch_size, dim=config.dim, depth=config.depth, num_heads=config.num_heads,
    #         num_inner_head=config.num_inner_head)

    # public net
    net = resnet34(pretrained=True)
    # net_weight_path = "./weight/resnet34-333f7ec4.pth"
    # assert os.path.exists(net_weight_path), "file {} does not exist.".format(net_weight_path)
    # net.load_state_dict(torch.load(net_weight_path, map_location='cpu'))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, config.class_num)

    net = net.to(device)
    net.train()

    if config.loss_function == 'CE':
        criterion = nn.CrossEntropyLoss().to(device)

    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    elif config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)

    if config.scheduler == 'cosine':
        lr_lambda = lambda epoch:(epoch*(1-config.warmup_decay)/config.warmup_epochs+config.warmup_decay) \
        if epoch < config.warmup_epochs else \
        (1-config.min_lr/config.lr)* 0.5 * (math.cos((epoch-config.warmup_epochs)/(config.epochs-config.warmup_epochs) * math.pi) + 1) + config.min_lr/config.lr
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1) 
    elif config.scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step, gamma=0.9)

    writer = SummaryWriter(comment='_'+config.net_name+'_'+config.writer_comment+'_'+str(fold))
    
    print("START TRAINING")
    best_acc=0
    ckpt_path = os.path.join(config.net_path, config.writer_comment, config.net_name)
    net_save_path = os.path.join(ckpt_path, str(fold))
    cm = None
    for epoch in range(config.epochs):
        y_true, y_pred = [], []
        cm = torch.zeros((config.class_num, config.class_num))
        epoch_loss = 0
        for i, pack in enumerate(train_loader):
            images = pack['imgs'].to(device)
            if images.shape[1] == 1:
                images = images.expand((-1, 3, -1, -1))
            labels = pack['labels'].to(device)
            names = pack['names']

            output = net(images)

            loss = criterion(output, labels)

            pred = output.argmax(dim=1)
            y_pred.append(pred)
            y_true.append(labels.tolist()[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            cm = confusion_matrix(pred.detach(), labels.detach(), cm)
        lr_scheduler.step()
        # cm = utils.calculate_confusion_matrix(y_pred, y_true, config.class_num)
        
        if (epoch + 1) % config.log_step == 0:
            print('[epoch %d]' % epoch)
            with torch.no_grad():
                result = valid(config, net, test_loader, criterion)
            val_loss, val_acc, sen, spe, auc, pre, f1score = result
            writer.add_scalar('Val/F1score', f1score, global_step=epoch)
            writer.add_scalar('Val/Pre', pre, global_step=epoch)
            writer.add_scalar('Val/Spe', spe, global_step=epoch)
            writer.add_scalar('Val/Sen', sen, global_step=epoch)
            writer.add_scalar('Val/AUC', auc, global_step=epoch)
            writer.add_scalar('Val/Acc', val_acc, global_step=epoch)
            writer.add_scalar('Val/Val_loss', val_loss, global_step=epoch)

            if epoch > config.epochs//4:
                if val_acc>best_acc:
                    best_acc=val_acc
                    print("=> saved best net")
                    if not os.path.exists(net_save_path):
                        os.makedirs(net_save_path)
                    if config.save_net:
                        torch.save(net.state_dict(), os.path.join(net_save_path, 'bestnet.pth'))
                    with open(os.path.join(net_save_path, 'result.txt'), 'w') as f:
                        f.write('Best Result:\n')
                        f.write('Acc: %f, Spe: %f, Sen: %f, AUC: %f, Pre: %f, F1score: %f'
                                % (val_acc, spe, sen, auc, pre, f1score))
        if epoch+1==config.epochs:
            with torch.no_grad():
                result = valid(config, net, test_loader, criterion)
            val_loss, val_acc, sen, spe, auc, pre, f1score = result
            if config.save_net:
                torch.save(net.state_dict(), os.path.join(net_save_path, 'last_epoch_net.pth'))
            with open(os.path.join(net_save_path, 'result.txt'), 'a') as f:
                f.write('\nLast Result:\n')
                f.write('Acc: %f, Spe: %f, Sen: %f, AUC: %f, Pre: %f, F1score: %f'
                        % (val_acc, spe, sen, auc, pre, f1score))
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        # print('Epoch [%d/%d], Avg Epoch Loss: %.4f' % (epoch + 1, config.epochs, avg_epoch_loss))
        writer.add_scalar('Train/LR', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
        writer.add_scalar('Train/Acc', cm.diag().sum() / cm.sum(), global_step=epoch)
        writer.add_scalar('Train/Avg_epoch_loss', avg_epoch_loss, global_step=epoch)


def seed_torch(seed=1):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    seed_torch(42)
    args = config()
    cv=KFold(n_splits=args.fold, random_state=42, shuffle=True)
    fold=0

    train_set = utils.get_dataset(args.data_path, args.csv_path, args.img_size, mode='train')
    test_set = utils.get_dataset(args.data_path, args.csv_path, args.img_size, mode='test')
    print(args)
    argspath = os.path.join(args.net_path, args.writer_comment, args.net_name)
    if not os.path.exists(argspath):
        os.makedirs(argspath)
    with open(os.path.join(argspath, 'net_info.txt'), 'w') as f:
        f.write(str(args))

    for train_idx,test_idx in cv.split(train_set):
        print("\nCross validation fold %d" %fold)
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler= SubsetRandomSampler(test_idx)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,sampler=train_sampler, num_workers=6)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False,sampler=test_sampler)
        train(args, train_loader, test_loader, fold, test_idx)
        fold+=1