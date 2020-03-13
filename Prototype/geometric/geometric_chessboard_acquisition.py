# -*- coding: utf-8 -*-
"""
Python script for geometric calibration of the prototype. The file open the device, take an image, downsample it and
then save it.
"""

if __name__ == "__main__":

    # Importation of standard modules
    import datetime
    import time
    import tkinter as tk
    import cv2 as cv2
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    matplotlib.use('TkAgg')

    # Importation of other modules
    import cameracontrol.cameracontrol as cc

    # Function
    def rawtorgb(im_dws, metadata):

        # Black level and saturation
        black_level = metadata["black_level"]
        im_dws -= black_level
        imtot = im_dws / (4096 - black_level)

        imtot = np.clip(imtot, 0, 1)
        imtot = np.clip((1.055 * imtot ** (1 / 2.4)) - 0.055, 0, 1) * 255
        imtot = imtot.astype(np.uint8)

        return cv2.cvtColor(imtot, cv2.COLOR_RGB2GRAY)

    # *** Code beginning *** ___________________________________________________________________________

    # Selection of folder to save image
    root = tk.Tk()
    root.withdraw()
    savingfolder = tk.filedialog.askdirectory(parent=root, initialdir="/", title='Please select a directory')
    root.destroy()

    # Take image instance
    camera_object = cc.TakeImage(imageformat="color24bits")

    exp_time = 15000  # microseconds
    binning = "2x2"  # CMOS binning
    image_number = 30  # Number of image, to be changed
    #image_number = 5
    timeinter = 7  # Time interval between each frame

    # Pre-allocation
    image_list = []

    band = ["R", "G", "B"]

    # Figure
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    for i in range(image_number):
        print("Image number {0} out of {1} ".format(i+1, image_number))
        for tim in range(timeinter, 0, -1):
                 time.sleep(1)
                 print("%02d seconds until next acquisition..." % tim)

        # Acquisition
        # Color image
        camera_object.imformat = camera_object.frmt_dict["color24bits"]
        im, metadata = camera_object.acquisition(exposuretime=exp_time, gain=0, binning=binning, video=False)

        gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        #gray_im = camera_object.dwnsampling(gray_im, "BGGR")  # Downsampling
        #gray_im = gray_im[:, :, 0].astype(np.uint8)

        print(gray_im.shape)

        # Raw image
        camera_object.imformat = camera_object.frmt_dict["raw"]
        im_raw, metadata_raw = camera_object.acquisition(exposuretime=exp_time, gain=0, binning=binning, video=False)

        im_dws = camera_object.dwnsampling(im_raw, "BGGR")
        im_raw_gray = rawtorgb(im_dws, metadata_raw)
        height_gray, width_gray = im_raw_gray.shape


        today = datetime.datetime.utcnow()

        #image = Image.fromarray(im, "RGB")
        #image = image.convert("L")
        #image.save(savingfolder + "/IMG_" + today.strftime("%Y%m%d_%H%M%S_UTC_") + ".png")

        today = datetime.datetime.utcnow()
        cv2.imwrite(savingfolder + "/IMG_" + today.strftime("%Y%m%d_%H%M%S_UTC_") + ".png", gray_im)

        #cv2.imshow("image", im_raw_gray)
        cv2.imshow("image", cv2.resize(gray_im, (int(gray_im.shape[1]/2), int(gray_im.shape[0]/2))))
        cv2.waitKey(1)

    cv2.destroyAllWindows()



        # # Downsampling
        # im_dws = camera_object.dwnsampling(im, "BGGR")
        #
        # # Black level
        # black_level = metadata["black_level"]
        # im_dws -= black_level
        #
        # today = datetime.datetime.utcnow()
        #
        # # Image total
        # imtot = im_dws/(4096 - black_level)
        # imtot = np.clip(imtot, 0, 1)
        #
        # grayim = camera_object.rgb2gray(imtot)
        # grayscale = 0.07 / grayim.mean()
        # imtot *= grayscale
        # imtot = np.clip(imtot, 0, 1)
        #
        # imtot = np.clip((1.055 * imtot ** (1 / 2.4)) - 0.055, 0, 1) * 255
        #
        # imtot = imtot.astype(np.uint8)
        #
        # image_total = Image.fromarray(imtot, "RGB")
        # image_total = image_total.convert("L")
        # image_total.save(savingfolder + "/IMG_" + today.strftime("%Y%m%d_%H%M%S_UTC_tot") + ".png")
        #
        # for n, b in enumerate(band):
        #
        #     im_pro = im_dws[:, :, n]/(4096 - black_level)
        #     im_pro = np.clip(im_pro, 0, 1)
        #
        #     brigthness = 0.07/im_pro.mean()
        #     im_pro *= brigthness
        #
        #     # Gamma
        #     imfin = (1.055 * im_pro ** (1 / 2.4)) - 0.055
        #
        #     imfin = np.clip(imfin, 0, 1) * 255
        #
        #     imfin = imfin.astype(np.uint8)
        #
        #     #  Visualization
        #     cv2.imshow("image", imfin)
        #     cv2.waitKey(2000)
        #
        #     image = Image.fromarray(imfin, "L")
        #     image.save(savingfolder + "/IMG_" + today.strftime("%Y%m%d_%H%M%S_UTC_") + b + ".png")

    camera_object.end()
