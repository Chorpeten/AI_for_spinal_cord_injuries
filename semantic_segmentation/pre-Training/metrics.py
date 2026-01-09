import numpy as np
import torch
import torch.nn.functional as F


# def iou_score(output, target):
#     smooth = 1e-5
#
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#     output_ = output > 0.5
#     target_ = target > 0.5
#     intersection = (output_ & target_).sum()
#     union = (output_ | target_).sum()
#
#     return (intersection + smooth) / (union + smooth)
def iou_score(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    gt = target.view(-1).data.cpu().numpy()
    output = output > 0.5
    gt = gt > 0.5
    output = output.reshape(-1)
    gt = gt.reshape(-1)
    TP = [i == 2 for i in np.sum([(output == 1.0), (gt == 1.0)], axis=0).tolist()]
    FN = [i == 2 for i in np.sum([(output == 0.0), (gt == 1.0)], axis=0).tolist()]
    FP = [i == 2 for i in np.sum([(output == 1.0), (gt == 0.0)], axis=0).tolist()]

    TP = float(np.sum(TP))
    FN = float(np.sum(FN))
    FP = float(np.sum(FP))

    # print('TP :{}'.format(TP))
    # print('FN :{}'.format(FN))
    # print('FP :{}'.format(FP))
    iou = (TP + smooth) / \
           (TP + FN + FP + smooth)
    # print('IOU: {}'.format(iou))
    return iou

# def dice_coef(output, target):
#     smooth = 1e-5
#
#     output = torch.sigmoid(output).view(-1).data.cpu().numpy()
#     target = target.view(-1).data.cpu().numpy()
#     intersection = (output * target).sum()
#
#     return (2. * intersection + smooth) / \
#         (output.sum() + target.sum() + smooth)

def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    gt = target.view(-1).data.cpu().numpy()
    output = output > 0.5
    gt = gt > 0.5
    output = output.reshape(-1)
    gt = gt.reshape(-1)
    TP = [i == 2 for i in np.sum([(output == 1.0), (gt == 1.0)], axis=0).tolist()]
    FN = [i == 2 for i in np.sum([(output == 0.0), (gt == 1.0)], axis=0).tolist()]
    FP = [i == 2 for i in np.sum([(output == 1.0), (gt == 0.0)], axis=0).tolist()]

    TP = float(np.sum(TP))
    FN = float(np.sum(FN))
    FP = float(np.sum(FP))

    # print('TP :{}'.format(TP))
    # print('FN :{}'.format(FN))
    # print('FP :{}'.format(FP))
    dice = (2 * TP + smooth) / \
           (TP + FN + TP + FP + smooth)
    # print('DICE: {}'.format(dice))
    return dice

def get_sensitivity(output, gt): # 求敏感度 se=TP/(TP+FN)
    SE = 0.
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    gt = gt.view(-1).data.cpu().numpy()
    output = output > 0.5
    gt = gt > 0.5
    output = output.reshape(-1)
    gt = gt.reshape(-1)
    TP = [i == 2 for i in np.sum([(output == 1.0), (gt == 1.0)],axis=0).tolist()]
    FN = [i == 2 for i in np.sum([(output == 0.0), (gt == 1.0)],axis=0).tolist()]
    #wfy:batch_num>1时，改进
    if len(output)>1:
        for i in range(len(output)):
            SE += float(np.sum(TP[i])) / (float(np.sum(TP[i]+FN[i])) + 1e-6)
    else:
        SE = float(np.sum(TP)) / (float(np.sum([TP,FN])) + 1e-6) #原本只用这一句
    SE = float(np.sum(TP)) / (float(np.sum([TP,FN])) + 1e-6)  # 原本只用这一句
    return SE  #返回batch中所有样本的SE和

def get_specificity(SR, GT, threshold=0.5):#求特异性 sp=TN/(FP+TN)
    SR = torch.sigmoid(SR).view(-1).data.cpu().numpy()
    GT = GT.view(-1).data.cpu().numpy()
    SR = SR > threshold
    SR = SR.reshape(-1)
    GT = GT > threshold
    GT = GT.reshape(-1)
    SP=0.# wfy
    # TN : True Negative
    # FP : False Positive
    # TN = ((SR == 0.0) + (GT == 0.0)) == 2
    TN = [i == 2 for i in np.sum([(SR == 0.0), (GT == 0.0)],axis=0).tolist()]
    # FP = ((SR == 1.0) + (GT == 0.0)) == 2
    FP = [i == 2 for i in np.sum([(SR == 1.0), (GT == 0.0)],axis=0).tolist()]
    # print(np.sum(TN))
    # print(np.sum([TN,FP]))

    #wfy:batch_num>1时，改进
    if len(SR)>1:
        for i in range(len(SR)):
            SP += float(np.sum(TN[i])) / (float(np.sum(TN[i] + FP[i])) + 1e-6)
    else:
        SP = float(np.sum(TN)) / (float(np.sum(TN + FP)) + 1e-6) # 原本只用这一句

    SP = float(np.sum(TN)) / (float(np.sum([TN,FP])) + 1e-6)
    return SP