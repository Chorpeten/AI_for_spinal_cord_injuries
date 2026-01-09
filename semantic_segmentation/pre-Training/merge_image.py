import PIL
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy

foreground = cv2.imread('./outputs/dsb2018_96_NestedUNet_woDS/0/p3_7.jpg')
background = cv2.imread('./inputs/dsb2018_96/images/p3_7.png')
masks = cv2.imread('./inputs/dsb2018_96/masks/0/p3_7.png')

# def ModifySize(_PicLocation):
#     im1 = cv2.imread(_PicLocation)
#     im2 = cv2.resize(im1, (256,256), inerpolation = cv2.INTER_CUBIC)
#     cv2.

fig, ax = plt.subplots(nrows=2, ncols=2)

ax[0][0].imshow(cv2.addWeighted(cv2.flip(foreground, 0), 1, cv2.flip(background, 0), 1, 0))
ax[0][0].set_title('Actual')
ax[1][0].imshow(cv2.addWeighted(cv2.flip(cv2.imread('./outputs/dsb2018_96_NestedUNet_woDS/0/p3_7.jpg'), 0), 1,
                                cv2.flip(cv2.imread('./inputs/dsb2018_96/images/p3_7.png'), 0), 1, 0))
ax[0][1].imshow(cv2.addWeighted(cv2.flip(masks, 0), 1, cv2.flip(background, 0), 1, 0))
ax[0][1].set_title('Prediction')
ax[1][1].imshow(cv2.addWeighted(cv2.flip(cv2.imread('./inputs/dsb2018_96/masks/0/p3_7.png'), 0), 1,
                                cv2.flip(cv2.imread('./inputs/dsb2018_96/images/p3_7.png'), 0), 1, 0))
plt.show()

