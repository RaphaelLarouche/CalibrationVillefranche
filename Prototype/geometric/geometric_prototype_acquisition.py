# -*- coding: utf-8 -*-
"""
Python file to perform geometric experiment acquisition.
"""

if __name__ == "__main__":

    # Importation of standard modules
    import matplotlib.pyplot as plt
    import datetime

    # Importation of other modules
    import cameracontrol.cameracontrol as cc

    # *** Code beginning ***
    camera_obj = cc.TakeImage(imageformat="raw")
    raw_image, metadata = camera_obj.acquisition(exposuretime=17000, gain=0, binning="2x2", video=False)
    camera_obj.end()

    # Saving image
    today = datetime.datetime.utcnow()
    imname = "/IMG_" + today.strftime("%Y%m%d_%H%M%S_UTC") + ".tif"   # year/month/day
    path_clef = "/Volumes/KINGSTON/Villefranche/Prototype/Geometric/"
    path = "geometric_proto_" + today.strftime("%Y%m%d") + imname
    camera_obj.saveTIFF_xiMU(path, raw_image, metadata)

    plt.figure()
    plt.imshow(raw_image)
    plt.show()
