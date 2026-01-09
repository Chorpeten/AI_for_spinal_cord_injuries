from nets.UCTransNet import UCTransNet as Net
import os
import numpy as np
import torch
import Config as config
import argparse
from PIL import Image
from torchvision import transforms
from cam_utils import *
import cv2
import matplotlib.pyplot as plt

#### 方法1：设置全局变量
parser = argparse.ArgumentParser(description='全局变量')
parser.add_argument('--resume',
                    default=r'/home/biocinformatic/Desktop/nowproject/UCTtransNet/SpinalCordInjury/UCTransNet_pretrain/Test_session_11.30_22h27/models/best_model-UCTransNet_pretrain.pth.tar',
                    help='加载的模型路径')
parser.add_argument('--img_src',
                    default='/home/biocinformatic/Desktop/nowproject/UCTtransNet/datasets/SpinalCordInjury/Test_filter/img/b_l.png',
                    help='单张图片src路径')
parser.add_argument('--img_gt',
                    default='/home/biocinformatic/Desktop/nowproject/UCTtransNet/SpinalCordInjury_visualize_test/b_l_predict.png',
                    help='单张图片gt路径')
# 其实该方法中并没有用到gt，是直接将预测图想作为loss回传，得到的梯度图像
args = parser.parse_args()

#### 方法2：API接口专用
model_pth = '/home/biocinformatic/Desktop/nowproject/UCTtransNet/SpinalCordInjury/UCTransNet_pretrain/Test_session_11.30_22h27/models/best_model-UCTransNet_pretrain.pth.tar'
# img_pth = 传参
# gt_pth = 传参

def main(model_pth = args.resume,img_pth = args.img_src, gt_pth = args.img_gt):
    # 1)建立模型、加载预训练参数
    config_vit = config.get_CTranS_config()
    # print(config_vit)
    model = Net(config_vit, n_channels=config.n_channels,
                n_classes=config.n_labels, vis=False)
    # model = Net()
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    # # 方法1调用
    # if os.path.exists(args.resume) and torch.cuda.is_available():
    #     print("=> 载入checkpoint'{}'".format(args.resume))
    #     checkpoint=torch.load(args.resume)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     print("=> checkpoint'{}'已载入".format(args.resume))
    # elif os.path.exists(args.resume) and not torch.cuda.is_available():
    #     print("=> 载入checkpoint'{}'".format(args.resume))
    #     checkpoint=torch.load(args.resume, map_location='cpu')
    #     model.load_state_dict(checkpoint['state_dict'])
    #     print("=> checkpoint'{}'已载入".format(args.resume))
    # else:
    #     print("=> 预训练模型路径出错'{}'".format(args.resume))
    # 方法2 API接口调用
    if os.path.exists(model_pth) and torch.cuda.is_available():
        # print("=> 载入checkpoint'{}'".format(model_pth))
        checkpoint=torch.load(model_pth)
        model.load_state_dict(checkpoint['state_dict'])
        # print("=> checkpoint'{}'已载入".format(model_pth))
    elif os.path.exists(model_pth) and not torch.cuda.is_available():
        # print("=> 载入checkpoint'{}'".format(model_pth))
        checkpoint=torch.load(model_pth, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        # print("=> checkpoint'{}'已载入".format(model_pth))
    else:
        print("=> 预训练模型路径出错'{}'".format(model_pth))
    # 2)传入图片，这里后期可以注释掉改为文件夹地址
    # # 方法1调用
    # # 展示预测结果，展示gt，预测结果与gt差值回传的grad_cam图，gt回传的grad_cam图
    # if os.path.exists(args.img_src) and os.path.exists(args.img_gt):
    #     # src = Image.open(args.img_src).convert('RGB')
    #     src = cv2.imread(args.img_src, 1)
    #     src = cv2.resize(src, (224, 224))
    #     src = np.array(src, dtype=np.uint8)
    #     # print(src.shape)
    #     # gt = Image.open(args.img_gt).convert('L')
    #     gt = cv2.imread(args.img_gt, 0)
    #     gt = cv2.resize(gt, (224, 224))
    #     gt = np.array(gt, dtype=np.uint8)
    #     gt = np.where((gt == 255)|(gt == 100), 1, 0) #变为双边缘
    #     # look(gt*255)
    # else:
    #     print("src地址：'{}'或gt地址：'{}'".format(args.img_src, args.img_gt))
    # 方法2 API接口调用
    # 展示预测结果，展示gt，预测结果与gt差值回传的grad_cam图，gt回传的grad_cam图
    if os.path.exists(img_pth) and os.path.exists(gt_pth):
        # src = Image.open(args.img_src).convert('RGB')
        src = cv2.imread(img_pth, 1)
        src = cv2.resize(src, (224, 224))
        src = np.array(src, dtype=np.uint8)
        # print(src.shape)
        # gt = Image.open(args.img_gt).convert('L')
        gt = cv2.imread(gt_pth, 0)
        gt = cv2.resize(gt, (224, 224))
        gt = np.array(gt, dtype=np.uint8)
        gt = np.where((gt == 255)|(gt == 100), 1, 0) #变为双边缘
        # look(gt*255)
    else:
        print("src地址：'{}'或gt地址：'{}'".format(img_pth, gt_pth))
    # 3)对图片进行预处理
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27))])
    # src和gt从ndarry变为tensor后，形状由[H，W，C]变为[C，H，W]
    src_tensor = data_transform(src)
    gt_tensor = transforms.ToTensor()(gt)

    # src和gt都增加一个维度
    src_tensor = torch.unsqueeze(src_tensor, dim=0) #[B_S, C, H, W]
    gt_tensor  = torch.unsqueeze(gt_tensor, dim=0)

    # 4)指定需要计算CAM的网络结构
    # target_layers = [model.up4]
    target_layers = [model.down4]
    # print(target_layers)

    # 5)调用Grad-CAM方法
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    grayscale_cam = cam(input_tensor=src_tensor, target=gt_tensor)
    grayscale_cam = grayscale_cam[0, :]
    # print(grayscale_cam.min())
    # print(grayscale_cam.max())
    visualization = show_cam_on_image(src.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    visualization_resize = cv2.resize(visualization, (384,384))
    return visualization_resize
    # plt.imshow(np.rot90(visualization_resize, 1))
    # plt.show()



if __name__=='__main__':
    main()
