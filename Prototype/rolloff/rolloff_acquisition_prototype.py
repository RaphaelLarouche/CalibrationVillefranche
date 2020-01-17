# -*- coding: utf-8 -*-
"""
Python file for the acquisition of image during roll-off experiment.
"""

if __name__ == "__main__":

    # Importation of standard modules
    import datetime
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    # Importation of other modules
    import cameracontrol.cameracontrol as cc


    # *** Beginning of the code ***

    # Open geometric calibration
    geocalib = np.load(os.path.dirname(
        os.getcwd()) + "/geometric/geometric_calibrationfiles_cb_air/geo_calibration_2x2_air_20191211_2152.npz")

    camera_object = cc.TakeImage(imageformat="raw")
    raw_image, metadata = camera_object.acquisition(exposuretime=20000, gain=0, binning="2x2", video=False)
    camera_object.end()


    # Saving image
    today = datetime.datetime.utcnow()
    imname = "/IMG_" + today.strftime("%Y%m%d_%H%M%S_UTC") + ".tif"
    #path = "/Volumes/KINGSTON/Villefranche/Prototype/Rolloff/rolloff_prototype_" + today.strftime("%Y%m%d") + imname
    path = "rolloff_proto_" + today.strftime("%Y%m%d") + imname
    camera_object.saveTIFF_xiMU(path, raw_image, metadata)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.imshow(raw_image)
    ax1.axhline(geocalib["centerpoint"][1], 0, 1300)
    ax1.axvline(geocalib["centerpoint"][0], 0, 1300)


    plt.show()
