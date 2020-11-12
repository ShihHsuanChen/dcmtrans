import sys
sys.path.append('..')

import numpy as np
from dcmtrans import dcmtrans
import PIL.Image


fname = 'D:\\dataset\\_kaggle_rsna_bone_age_short\\test\\4362.png'
img0 = PIL.Image.open(fname)
# img0.show()

image_arr = np.array(img0)
# PIL.Image.fromarray(image_arr, 'L').show()

image_arr = np.expand_dims(image_arr, 0)
print(image_arr.shape)
img1 = dcmtrans.image_convert(image_arr)
# img1.show()

image_arr = np.concatenate((
    image_arr,
    0*np.ones(image_arr.shape),
    0*np.ones(image_arr.shape),
    128*np.ones(image_arr.shape),
    ), axis=0)
print(image_arr.shape)
img2 = dcmtrans.image_convert(image_arr, fmt='RGBA')
img2.show()
# img2.save('D:\\test.png')
