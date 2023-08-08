import os
import cv2
from skimage.measure import compare_ssim as ssim
from sklearn.metrics import mean_squared_error




def calculate_metrics(pre,tre):
    ssim_score=ssim(pre[tre>90], tre[tre>90])  # for calculate only the pixels with the background area removed
    mean_error_score=mean_squared_error(pre[tre>90]/255.0, tre[tre>90]/255.0)  # for calculate only the pixels with the background area removed
    dice_score= (2 * ((pre/255.0) * (tre/255.0)).sum() + 0.001) / ((pre/255.0).sum() + (tre/255.0).sum())
    return ssim_score,mean_error_score,dice_score


pre_image=cv2.imread('./image/pre.png')[:,:,0]
gt_image=cv2.imread('./image/tre.tiff')[:,:, 0]

ssim_score,mean_error_score,dice_score=calculate_metrics(pre_image,gt_image)
print('ssim_score:',ssim_score)
print('mean_error_score:',mean_error_score)
print('dice_score:',dice_score)

