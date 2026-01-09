import os
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm, trange
from Grad_cam_main import *

# foreground = cv2.imread('./outputs/dsb2018_96_NestedUNet_woDS/0/p3_7.jpg')
# background = cv2.imread('./inputs/dsb2018_96/images/p3_7.png')
# masks = cv2.imread('./inputs/dsb2018_96/masks/0/p3_7.png')

background_pth = glob.glob(os.path.join('datasets/SpinalCordInjury/Test_filter/img/', '**.png'))
foreground_pth = glob.glob(os.path.join('SpinalCordInjury_visualize_test/','**_predict.png'))
masks_pth = glob.glob(os.path.join('SpinalCordInjury_visualize_test/','**_lab.png'))

background_pth, foreground_pth, masks_pth = sorted(background_pth), sorted(foreground_pth), sorted(masks_pth)
save_pth = './Result_combine/'
if not os.path.exists(save_pth):
    os.makedirs(save_pth)
model_pth = '/home/biocinformatic/Desktop/nowproject/UCTtransNet/SpinalCordInjury/UCTransNet_pretrain/Test_session_11.30_22h27/models/best_model-UCTransNet_pretrain.pth.tar'
# print(background_pth[0].split('/')[-1])
for i in tqdm(range(len(background_pth))):
    plt.figure(figsize=(24,20), dpi=90)
    plt.subplot(2,2,1)
    figure_original = np.rot90(cv2.imread(background_pth[i]))
    plt.imshow(figure_original)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Original',fontsize=26)
    plt.subplot(2,2,2)
    figure_actual = np.rot90(cv2.resize(cv2.imread(masks_pth[i]), (384,384)), 1)
    figure_actual[:,:,0][figure_actual[:,:,0] > 0] = 0
    figure_actual[:,:,1][figure_actual[:,:,1] > 0] = 0
    figure_actual[:,:,2][figure_actual[:,:,2] > 0] = 200
    combine_actual = cv2.addWeighted(figure_actual, 1, np.rot90(cv2.imread(background_pth[i]), 1), 1, 0)
    plt.imshow(combine_actual)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Actual',fontsize=26)
    plt.subplot(2,2,3)
    figure_predict = np.rot90(cv2.resize(cv2.imread(foreground_pth[i]), (384,384)), 1)
    figure_predict[:,:,0][figure_predict[:,:,0] > 0] = 150
    figure_predict[:,:,1][figure_predict[:,:,1] > 0] = 0
    figure_predict[:,:,2][figure_predict[:,:,2] > 0] = 0
    combine_predict = cv2.addWeighted(figure_predict, 1, np.rot90(cv2.imread(background_pth[i]), 1), 1, 0)
    plt.imshow(combine_predict)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Predict',fontsize=26)
    plt.subplot(2, 2, 4)
    plt.imshow(np.rot90(main(model_pth, background_pth[i], foreground_pth[i]),1))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Attention',fontsize=26)
    plt.savefig(save_pth+background_pth[i].split('/')[-1].split('.')[0]+'.pdf')
    # plt.show()

# plt.subplot(1,2,1)
# figure_actual = np.rot90(cv2.resize(cv2.imread(masks_pth[0]), (384,384)), 1)
# figure_actual[:,:,0][figure_actual[:,:,0] > 0] = 0
# figure_actual[:,:,1][figure_actual[:,:,1] > 0] = 0
# figure_actual[:,:,2][figure_actual[:,:,2] > 0] = 200
# combine_actual = cv2.addWeighted(figure_actual, 1, np.rot90(cv2.imread(background_pth[0]), 1), 1, 0)
# plt.imshow(combine_actual)
# plt.title('Actual')
# plt.subplot(1,2,2)
# figure_predict = np.rot90(cv2.resize(cv2.imread(foreground_pth[0]), (384,384)), 1)
# figure_predict[:,:,0][figure_predict[:,:,0] > 0] = 150
# figure_predict[:,:,1][figure_predict[:,:,1] > 0] = 0
# figure_predict[:,:,2][figure_predict[:,:,2] > 0] = 0
# combine_predict = cv2.addWeighted(figure_predict, 1, np.rot90(cv2.imread(background_pth[0]), 1), 1, 0)
# plt.imshow(combine_predict)
# plt.title('Predict')
# plt.show()
# fig, ax = plt.subplots(nrows=2, ncols=2)
#
# ax[0][0].imshow(cv2.addWeighted(cv2.flip(foreground, 0), 1, cv2.flip(background, 0), 1, 0))
# ax[0][0].set_title('Actual')
# ax[1][0].imshow(cv2.addWeighted(cv2.flip(cv2.imread('./outputs/dsb2018_96_NestedUNet_woDS/0/p3_7.jpg'), 0), 1,
#                                 cv2.flip(cv2.imread('./inputs/dsb2018_96/images/p3_7.png'), 0), 1, 0))
# ax[0][1].imshow(cv2.addWeighted(cv2.flip(masks, 0), 1, cv2.flip(background, 0), 1, 0))
# ax[0][1].set_title('Prediction')
# ax[1][1].imshow(cv2.addWeighted(cv2.flip(cv2.imread('./inputs/dsb2018_96/masks/0/p3_7.png'), 0), 1,
#                                 cv2.flip(cv2.imread('./inputs/dsb2018_96/images/p3_7.png'), 0), 1, 0))
# plt.show()

