# -*- coding: utf-8 -*-
"""
Python file for roll-off data analysis for the prototype with 2x2 binning (integrating sphere).
"""

if __name__ == "__main__":

    # Importation of standard modules
    import numpy as np
    import glob
    import matplotlib.pyplot as plt

    # Importation of other modules
    import cameracontrol.cameracontrol as cc

    # Function

    # *** Code beginning ***
    processing = cc.ProcessImage()

    # Image directory creation
    path_00 = "/Volumes/KINGSTON/Quebec/Prototype/Rolloff/rolloff_proto_air/rolloff_proto_20191213_2x2"
    images_path_00 = glob.glob(path_00 + "/IMG_*.tif")
    images_path_00.sort()

    # Opening dark image
    image_path_dark = glob.glob(path_00 + "/DARK*.tif")
    image_path_dark.sort()
    imdark_00, metdark_00 = processing.readTIFF_xiMU(image_path_dark[0])

    # ___________________________________________________________________________
    # Loop to get roll=off curves

    # Pre-allocation
    imtotal_00 = np.zeros((imdark_00.shape[0] // 2, imdark_00.shape[1] // 2, 3))

    centroids_00 = np.empty((len(images_path_00), 3), dtype=[("y", "float32"), ("x", "float32")])
    centroids_00.fill(np.nan)

    rolloff_00 = np.empty((len(images_path_00), 3))
    rolloff_00.fill(np.nan)

    angles = np.arange(-100, 105, 5)

    fig1, ax1 = plt.subplots(1, 3)

    for n, path in enumerate(images_path_00):
        print("Processing image number {0}".format(n))

        # Reading data
        im, met = processing.readTIFF_xiMU(path)
        im -= imdark_00

        im_dws = processing.dwnsampling(im, "BGGR", ave=True)

        imtotal_00 += im_dws

        for i in range(im_dws.shape[2]):

            # Centroids
            bin_im, regprops = processing.regionproperties(im_dws[:, :, i], 0.3E3)

            if regprops:
                centroids_00[n, i] = regprops[0].centroid
                yc, xc = regprops[0].centroid
                yc = int(round(yc))
                xc = int(round(xc))
                ax1[i].plot(xc, yc, "r+")

                rolloff_00[n, i] = np.mean(im_dws[yc, xc-1:xc+2:1, i])

    # Printing results
    print(rolloff_00)

    rolloff_00 = rolloff_00 / np.nanmax(rolloff_00, axis=0)
    print(rolloff_00)

    # Cliping image
    imtotal_00 = np.clip(imtotal_00, 0, 2**12)

    # Figure configuration
    # Figure 1
    for j in range(imtotal_00.shape[2]):
        ax1[j].imshow(imtotal_00[:, :, j])

    # Figure 2
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    ax2.plot(angles, rolloff_00[:, 0], label="Red channel")
    ax2.plot(angles, rolloff_00[:, 1], label="Green channel")
    ax2.plot(angles, rolloff_00[:, 2], label="Blue channel")

    ax2.set_xlabel("Angles [˚]")
    ax2.set_ylabel("Roll-off")

    ax2.legend(loc="best")

    plt.show()
