import numpy as np
import matplotlib.pyplot as plt


class SatPlotter:

    """
    Simple plotting tools for visualizing results of USatellite.  May want to reuse as we vary 
    experiments and save various models and their weights.
    """

    def plot_image_and_result(self, img, val, result):

        """
        Straight-forward plot of a result using a model with only the input image (no other engineered features).
        """

        ind = np.random.randint(len(img))

        r_band = (img[ind,:,:,0]-np.min(img[ind,:,:,0]))/(np.max(img[ind,:,:,0])-np.min(img[ind,:,:,0]))
        g_band = (img[ind,:,:,1]-np.min(img[ind,:,:,1]))/(np.max(img[ind,:,:,1])-np.min(img[ind,:,:,1]))
        b_band = (img[ind,:,:,2]-np.min(img[ind,:,:,2]))/(np.max(img[ind,:,:,2])-np.min(img[ind,:,:,2]))
        RGB = np.stack((r_band, g_band, b_band), axis=-1)

        val = np.argmax(val, axis=-1)

        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 50))
        im0 = ax[0].imshow(RGB)
        ax[0].set_title('Remote Sensing Image', fontsize=30)
        im1 = ax[1].imshow(val[ind], vmin=0, vmax=5)
        ax[1].set_title('Validation Image', fontsize=30)
        im2 = ax[2].imshow(result[ind], vmin=0, vmax=5)
        ax[2].set_title('Classified Image', fontsize=30)

    def plot_probabilities(self, img, labels, probabilities):

        """
        Plots images, labels, and prediction probabilities per class.  Intended for QCing validation data.
        """

        ind = np.random.randint(len(img))

        r_band = (img[ind,:,:,0]-np.min(img[ind,:,:,0]))/(np.max(img[ind,:,:,0])-np.min(img[ind,:,:,0]))
        g_band = (img[ind,:,:,1]-np.min(img[ind,:,:,1]))/(np.max(img[ind,:,:,1])-np.min(img[ind,:,:,1]))
        b_band = (img[ind,:,:,2]-np.min(img[ind,:,:,2]))/(np.max(img[ind,:,:,2])-np.min(img[ind,:,:,2]))
        RGB = np.stack((r_band, g_band, b_band), axis=-1)

        labels = np.argmax(labels, axis=-1)

        fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(50, 50))
        im0 = ax[0,0].imshow(RGB)
        ax[0,0].set_title('Remote Sensing Image', fontsize=30)
        im1 = ax[0,1].imshow(labels[ind], vmin=0, vmax=5)
        ax[0,1].set_title('Label', fontsize=30)
        im3 = ax[1,0].imshow(probabilities[ind,:,:,0], vmin=0, vmax=1)
        ax[1,0].set_title('Class 1', fontsize=30)
        im4 = ax[1,1].imshow(probabilities[ind,:,:,1], vmin=0, vmax=1)
        ax[1,1].set_title('Class 2', fontsize=30)
        im5 = ax[1,2].imshow(probabilities[ind,:,:,2], vmin=0, vmax=1)
        ax[1,2].set_title('Class 3', fontsize=30)
        im6 = ax[2,0].imshow(probabilities[ind,:,:,3], vmin=0, vmax=1)
        ax[2,0].set_title('Class 4', fontsize=30)
        im7 = ax[2,1].imshow(probabilities[ind,:,:,4], vmin=0, vmax=1)
        ax[2,1].set_title('Class 5', fontsize=30)
        im8 = ax[2,2].imshow(probabilities[ind,:,:,5], vmin=0, vmax=1)
        ax[2,2].set_title('Class 6', fontsize=30)

    def plot_img_augs(self, img, label):

        """
        Straight-forward plot of a result using a model with only the input image (no other engineered features).
        """

        ind = np.random.randint(len(img))

        r_band = (img[ind,:,:,0]-np.min(img[ind,:,:,0]))/(np.max(img[ind,:,:,0])-np.min(img[ind,:,:,0]))
        g_band = (img[ind,:,:,1]-np.min(img[ind,:,:,1]))/(np.max(img[ind,:,:,1])-np.min(img[ind,:,:,1]))
        b_band = (img[ind,:,:,2]-np.min(img[ind,:,:,2]))/(np.max(img[ind,:,:,2])-np.min(img[ind,:,:,2]))
        RGB = np.stack((r_band, g_band, b_band), axis=-1)

        label = np.argmax(label, axis=-1)

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(20, 50))
        im0 = ax[0].imshow(RGB)
        ax[0].set_title('Remote Sensing Image', fontsize=30)
        im1 = ax[1].imshow(label[ind], vmin=0, vmax=5)
        ax[1].set_title('Labeled image', fontsize=30)
 