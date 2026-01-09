import glob
from tqdm import tqdm
import pandas as pd
import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from nets.UCTransNet import UCTransNet
from utils import *
import time
import cv2
import pandas
import metrics

def show_image_with_dice(predict_save, labs, save_path):
    # tmp_lbl = (labs).astype(np.float32)
    # tmp_3dunet = (predict_save).astype(np.float32)
    # dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    # # dice_show = "%.3f" % (dice_pred)
    # iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    dice_pred = metrics.dice_coef(predict_save, labs)
    iou_pred = metrics.iou_score(predict_save, labs)
    specificity_pred = metrics.get_specificity(predict_save, labs)
    senseticity_pred = metrics.get_sensitivity(predict_save,labs)
    # print('\nSpecificity:%s'%specificity_pred)
    # print('\nSenseticity:%s' % senseticity_pred)

    return dice_pred, iou_pred,specificity_pred,senseticity_pred

def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    model.eval()

    output = model(input_img.cuda())
    pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp,specificity_pred,senseticity_pred = show_image_with_dice(predict_save, labs, save_path=vis_save_path+'_predict'+'.png')
    return dice_pred_tmp, iou_tmp,specificity_pred,senseticity_pred


if __name__ == '__main__':
    test_list = glob.glob(os.path.join('SpinalCordInjury/UCTransNet_train/','**'))
    test_session = [k.split('/')[-1] for k in test_list]

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    test_num = 89
    model_type = config.model_name

    Time, mIOU, mDICE, Specificity, Sensitivity = [], [], [], [], []
    for m in range(len(test_session)):
        save_path  = config.task_name +'/'+ model_type +'/' + test_session[m] + '/'
        # vis_path = "./" + config.task_name + '_visualize_test/'
        vis_path = "./" + config.task_name + 'test/'
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)

        model_path = "./SpinalCordInjury/UCTransNet_train/{}/models/best_model-UCTransNet_pretrain.pth.tar".format(
            test_session[m])
        checkpoint = torch.load(model_path, map_location='cuda')

        if model_type == 'UCTransNet':
            config_vit = config.get_CTranS_config()
            model = UCTransNet(config_vit,n_channels=config.n_channels,n_classes=config.n_labels)

        elif model_type == 'UCTransNet_pretrain':
            config_vit = config.get_CTranS_config()
            model = UCTransNet(config_vit,n_channels=config.n_channels,n_classes=config.n_labels)

        else: raise TypeError('Please enter a valid name for the model type')

        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
            model = nn.DataParallel(model, device_ids=[0,1,2,3])
        model.load_state_dict(checkpoint['state_dict'])
        print('Model loaded !')
        tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
        test_dataset = ImageToImage2D(config.test_dataset, tf_test,image_size=config.img_size)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        # test_num = int(len(test_dataset.images_list))
        # print(len(test_dataset.images_list))

        dice_pred = 0.0
        iou_pred = 0.0
        dice_ens = 0.0
        specificity_pred = 0.0
        sensitivity_pred = 0.0
        start_time = time.time()

        with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
            for i, (sampled_batch, names) in enumerate(test_loader, 1):
                test_data, test_label = sampled_batch['image'], sampled_batch['label']
                arr=test_data.numpy()
                arr = arr.astype(np.float32())
                lab=test_label.data.numpy()
                img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
                # fig, ax = plt.subplots()
                # plt.imshow(img_lab, cmap='gray')
                # plt.axis("off")
                # height, width = config.img_size, config.img_size
                # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
                # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                # plt.margins(0, 0)
                # plt.savefig(vis_path+str(names).split('.')[0].split('\'')[-1]+"_lab.png", dpi=300)
                # plt.close()
                input_img = torch.from_numpy(arr)
                dice_pred_t,iou_pred_t,specifi_t,senseti_t = vis_and_save_heatmap(model, input_img, None, lab,
                                                              vis_path+str(names).split('.')[0].split('\'')[-1],
                                                   dice_pred=dice_pred, dice_ens=dice_ens)


                dice_pred+=dice_pred_t
                iou_pred+=iou_pred_t
                specificity_pred+=specifi_t
                sensitivity_pred+=senseti_t
                torch.cuda.empty_cache()
                pbar.update()
        end_time = time.time()
        mDICE.append(dice_pred/test_num)
        mIOU.append(iou_pred/test_num)
        Time.append((end_time - start_time)/test_num)
        Specificity.append(specificity_pred / test_num)
        Sensitivity.append(sensitivity_pred / test_num)

        print('-' * 20)
        print('Model {} completed !'.format(m + 1))
        print ("dice_pred",dice_pred/test_num)
        print ("iou_pred",iou_pred/test_num)
        print('Total time:{}'.format(end_time-start_time))
        print('Mean time:{}'.format((end_time - start_time)/test_num))
        print ("specificity_pred", specificity_pred / test_num)
        print ("sensitivity_pred", sensitivity_pred / test_num)

    df = pd.DataFrame({
            'Time':Time,
            'IOU':mIOU,
            'DICE':mDICE,
            'Specificity':Specificity,
            'Sensitivity':Sensitivity
        })
    print(df)
    # df.to_csv('K5_mean_metircs.csv')







