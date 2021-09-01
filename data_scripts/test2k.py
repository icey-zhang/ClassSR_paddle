import os
import cv2
import sys
from scipy import ndimage
import paddle.nn.functional as F
import paddle
sys.path.append("..")
import data.util as util
num = 1301 #1201-1300
#we first downsample the original images with scaling factors 0.6, 0.7, 0.8, 0.9 to generate the HR/LR images.
for scale in [0.25]:
    GT_folder = '/home/aistudio/data/data55117/test8k'#'/data0/xtkong/data/DIV2K800_GT' #数据读取路径 
    save_GT_folder = '/home/aistudio/test2K/GT' #数据保存路径  test2K/GT 不需要修改
    save_LR_folder = '/home/aistudio/test2K/LR' #数据保存路径  test2K/LR 不需要修改
    for i in [save_GT_folder]:
        if os.path.exists(i):
            pass
        else:
            os.makedirs(i)
    for i in [save_LR_folder]:
        if os.path.exists(i):
            pass
        else:
            os.makedirs(i)
    img_GT_list = util._get_paths_from_images(GT_folder)

    for path_GT in img_GT_list:
        img_GT = cv2.imread(path_GT)
        img_GT = img_GT
        # imresize

        # rlt_GT = util.imresize_np(img_GT, scale, antialiasing=True)
        #print(str(scale) + "_" + os.path.basename(path_GT))
        # rlt_GT = util.imresize_np(img_GT, scale, antialiasing=True)
        H,W,C = img_GT.shape
        # img = paddle.to_tensor(img_GT,dtype='float32')
        # img = paddle.expand(img, shape=[1,H,W,C])
        # rlt_GT = F.upsample(x=img, size=[H*scale,W*scale],mode='BICUBIC',data_format='NHWC')
        # rlt_HR = F.upsample(x=rlt_GT, size=[H*scale*0.25,W*scale*0.25],mode='BICUBIC',data_format='NHWC')

        # rlt_GT = rlt_GT.squeeze().numpy()
        # rlt_LR = rlt_HR.squeeze().numpy()
        rlt_GT = util.imresize_np(img_GT,0.25, antialiasing=True)
        rlt_LR = util.imresize_np(img_GT,0.25*0.25, antialiasing=True)
        # print(img_GT.shape)
        # print(rlt_GT.shape)
        # print(rlt_LR.shape)
        print(os.path.basename(path_GT).split('.')[0])
        if eval(os.path.basename(path_GT).split('.')[0])<num:
            cv2.imwrite(os.path.join(save_GT_folder, os.path.basename(path_GT)), rlt_GT)
            cv2.imwrite(os.path.join(save_LR_folder, os.path.basename(path_GT)), rlt_LR)
        else:
            break
        
