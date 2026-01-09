import numpy as np
import torch

def iou_score(output, target):
    smooth = 1e-5

    gt = target
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

def dice_coef(output, target):
    smooth = 1e-5

    gt = target
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


def get_sensitivity(output, gt):
    SE = 0.
    output = output > 0.5
    gt = gt > 0.5
    output = output.reshape(-1)
    gt = gt.reshape(-1)
    TP = [i == 2 for i in np.sum([(output == 1.0), (gt == 1.0)], axis=0).tolist()]
    FN = [i == 2 for i in np.sum([(output == 0.0), (gt == 1.0)], axis=0).tolist()]

    if len(output) > 1:
        for i in range(len(output)):
            SE += float(np.sum(TP[i])) / (float(np.sum(TP[i] + FN[i])) + 1e-6)
    else:
        SE = float(np.sum(TP)) / (float(np.sum([TP, FN])) + 1e-6)
    SE = float(np.sum(TP)) / (float(np.sum([TP, FN])) + 1e-6)
    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = SR > threshold
    SR = SR.reshape(-1)
    GT = GT > threshold
    GT = GT.reshape(-1)
    SP = 0.
    # TN : True Negative
    # FP : False Positive
    # TN = ((SR == 0.0) + (GT == 0.0)) == 2
    TN = [i == 2 for i in np.sum([(SR == 0.0), (GT == 0.0)], axis=0).tolist()]
    # FP = ((SR == 1.0) + (GT == 0.0)) == 2
    FP = [i == 2 for i in np.sum([(SR == 1.0), (GT == 0.0)], axis=0).tolist()]
    # print(np.sum(TN))
    # print(np.sum([TN,FP]))

    if len(SR) > 1:
        for i in range(len(SR)):
            SP += float(np.sum(TN[i])) / (float(np.sum(TN[i] + FP[i])) + 1e-6)
    else:
        SP = float(np.sum(TN)) / (float(np.sum(TN + FP)) + 1e-6)  # 原本只用这一句

    SP = float(np.sum(TN)) / (float(np.sum([TN, FP])) + 1e-6)
    return SP