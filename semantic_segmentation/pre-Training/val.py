import argparse
import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import yaml
import albumentations as A
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import archs
from dataset import Dataset
from metrics import iou_score,dice_coef,get_specificity,get_sensitivity
from utils import AverageMeter
import time
import random

"""
需要指定参数：--name dsb2018_96_NestedUNet_woDS
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='Spinal_Cord_Injury',
                        help='model name')

    args = parser.parse_args()

    return args


def main(model_num = 0):
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['name'] = 'Spinal_Cord_Injury'

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', 'test_data', 'images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    val_transform = Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', 'test_data', 'images'),
        mask_dir=os.path.join('inputs', 'test_data', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()
    dice_meter = AverageMeter()
    speci_meter = AverageMeter()
    senseti_meter = AverageMeter()

    model.load_state_dict(torch.load(model_list[model_num]))
    model.eval()
    start_time = time.time()
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            specificity = get_specificity(output, target)
            sensitivity = get_sensitivity(output, target)

            avg_meter.update(iou, input.size(0))
            dice_meter.update(dice, input.size(0))
            speci_meter.update(specificity, input.size(0))
            senseti_meter.update(sensitivity, input.size(0))

            # output = torch.sigmoid(output).cpu().numpy()
            #
            # for i in range(len(output)):
            #     for c in range(config['num_classes']):
            #         cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
            #                     (output[i, c] * 255).astype('uint8'))
    end_time = time.time()

    print('Total time:%s'%(end_time-start_time))
    print('Mean time:%s'%((end_time-start_time)/len(val_img_ids)))
    print('IoU: %.4f' % avg_meter.avg)
    print('DICE: %.4f' % dice_meter.avg)
    print('Specificity: %.4f' % speci_meter.avg)
    print('Sensitivity: %.4f' % senseti_meter.avg)
    return (end_time-start_time)/len(val_img_ids),avg_meter.avg,dice_meter.avg,speci_meter.avg,senseti_meter.avg

model_list = glob(os.path.join('./models/Spinal_Cord_Injury','**.pth'))

if __name__ == '__main__':
    Time,mIOU,mDICE,Specificity,Sensitivity = [],[],[],[],[]
    for i in range(5):
        all_time_start = time.time()
        m_time, miou, mdice, mspf, msen = main(model_num=i)
        all_time_end = time.time()
        all_time = all_time_end - all_time_start
        Time.append(m_time)
        mIOU.append(miou)
        mDICE.append(mdice)
        Specificity.append(mspf)
        Sensitivity.append(msen)
        print('Model {} completed !'.format(i+1))
        print('Spend time {}'.format(all_time))

    df = pd.DataFrame({
        'Time':Time,
        'IOU':mIOU,
        'DICE':mDICE,
        'Specificity':Specificity,
        'Sensitivity':Sensitivity
    })
    print(df)
    df.to_csv('K5_mean_metrics.csv')


