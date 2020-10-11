import numpy as np
from skimage.transform import rotate, AffineTransform
from skimage import transform
from skimage.util import random_noise

class Augmentor:

    def img_augs(imgs_array, label_array, num_examples):

        AFT = AffineTransform(shear=0.5)

        new_imgs = np.zeros((num_examples,512,512,3))
        new_labels = np.zeros((num_examples,512,512,6))

        for i in range(num_examples):

            idx = np.random.randint(0,imgs_array.shape[0])

            auged_img = imgs_array[idx]
            auged_label = label_array[idx]

            if np.random.random()>=0.5:

                ### Positional augmentations first.  This modified both images and labels.

                if np.random.random()>=0.5:
                    auged_img = np.fliplr(auged_img)
                    auged_label = np.fliplr(auged_label)

                if np.random.random()>=0.5:
                    auged_img = np.flipud(auged_img)
                    auged_label = np.flipud(auged_label)

                if np.random.random()>=0.5:
                    rot = np.random.randint(0,90)
                    auged_img = rotate(auged_img, angle=rot)
                    auged_label = rotate(auged_label, angle=rot)

                if np.random.random()>=0.5:
                    auged_img = transform.warp(auged_img, AFT, order=1, preserve_range=True, mode='wrap')
                    auged_label = transform.warp(auged_label, AFT, order=1, preserve_range=True, mode='wrap')

                ### Color augmentations are only applied to the images.

                if np.random.random()>=0.5:
                    auged_img = random_noise(auged_img, var=0.1**2)

                if np.random.random()>=0.5:
                    auged_img = auged_img + (100/255)

                if np.random.random()>=0.5:
                    if np.random.random()>=0.75:
                        auged_img = auged_img*1.5
                    else: auged_img = auged_img/1.5

            else:

                auged_img = auged_img
                auged_label = auged_label

            new_imgs[i] = auged_img
            new_labels[i] = auged_label

        return new_imgs, new_labels