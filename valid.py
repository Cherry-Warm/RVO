import torch
from utils import confusion_matrix, calculate_confusion_matrix, cal_TPNF
from sklearn.metrics import roc_auc_score

def valid(config, net, val_loader, criterion):                                       
    device = next(net.parameters()).device
    net.eval()

    print("START VALIDING")
    epoch_loss = 0
    y_true, y_score = [], []

    cm = torch.zeros((config.class_num, config.class_num))
    for i, pack in enumerate(val_loader):
        images = pack['imgs'].to(device)
        if images.shape[1] == 1:
            images = images.expand((-1, 3, -1, -1))
        names = pack['names']
        labels = pack['labels'].to(device)

        output = net(images)

        loss = criterion(output, labels)

        pred = output.argmax(dim=1)
        y_true.append(labels.detach().item())
        y_score.append(output[0].softmax(0)[1].item())

        cm = confusion_matrix(pred.detach(), labels.detach(), cm)
        epoch_loss += loss.cpu()
            
    avg_epoch_loss = epoch_loss / len(val_loader)
    # cm = calculate_confusion_matrix(y_score, y_true, config.class_num)

    #   t  N  P
    # p
    # N    TN FN
    # P    FP TP

    TP, FN, TN, FP = cal_TPNF(cm)

    # acc = cm.diag().sum() / cm.sum()
    acc = (TP + TN) / cm.sum()
    sen = TP / (TP + FN)
    spe = TN / (TN + FP)
    # pre = cm.diag()[1] / (cm.sum(dim=1) + 1e-6)[1]
    pre = TP / (TP + FP)
    rec = sen
    f1score = 2*pre*rec / (pre+rec+ 1e-6)
    # auc = roc_auc_score(y_true, y_score)
    auc = (sen + spe) / 2
    
    return [avg_epoch_loss, acc, sen, spe, auc, pre, f1score]