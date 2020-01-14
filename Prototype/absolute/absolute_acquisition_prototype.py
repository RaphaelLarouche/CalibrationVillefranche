# -*- coding: utf-8 -*-
"""
Python file to perform absolute experiment acquisition.
"""

if __name__ == "__main__":

    # Importation of standard modules
    import matplotlib.pyplot as plt
    import datetime
    import time

    # Importation of other modules
    import cameracontrol.cameracontrol as cc

    # *** Code beginning ***
    camera_obj = cc.TakeImage(imageformat="raw")
    exp = 17000
    bin = "2x2"
    number_im = 1  # Number of image, to be changed
    for i in range(number_im):
        time.sleep(20)
        raw_image, metadata = camera_obj.acquisition(exposuretime=exp, gain=0, binning=bin, video=False)
        # Saving image
        today = datetime.datetime.utcnow()
        imname = "/IMG_" + today.strftime("%Y%m%d_%H%M%S_UTC") + ".tif"  # year/month/day
        path = "absolute_proto_" + today.strftime("%Y%m%d") + "_{0}_{1}us".format(bin, exp) + imname
        camera_obj.saveTIFF_xiMU(path, raw_image, metadata)
        #time.sleep(2)
    camera_obj.end()

    plt.figure()
    plt.imshow(raw_image)
    plt.show()

