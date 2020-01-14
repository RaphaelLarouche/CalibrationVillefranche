# -*- coding: utf-8 -*-
"""
Python file to perform roll-off experiment analysis.
"""

if __name__ == "__main__":

    # Importation of standard modules
    import matplotlib.pyplot as plt
    import numpy as np
    import glob
    import os
    import statistics

    # Importation of other modules
    import cameracontrol.cameracontrol as cc

    # *** Code beginning ***

    # ProcessImage instance from cameracontrol
    processing = cc.ProcessImage()

    # Choosing binning
    while True:
        answer = input("Which binning do you want to analyze? (4/2): ")
        if answer in ["4", "2"]:
            break
    # 4x4
    #impath_00 = "/Volumes/KINGSTON/Villefranche/Prototype/Rolloff/rolloff_proto_20191114_4x4_00"
    impath_00 = "/Volumes/KINGSTON/Villefranche/Prototype/Rolloff/rolloff_proto_20191127_4x4_00_02"
    impath_90 = "/Volumes/KINGSTON/Villefranche/Prototype/Rolloff/rolloff_proto_20191128_4x4_90_01"

    if answer == "2":
        # 2x2
        impath_00 = "/Volumes/KINGSTON/Villefranche/Prototype/Rolloff/rolloff_proto_20191127_2x2_00_03"
        #impath_00 = "/Volumes/KINGSTON/Villefranche/Prototype/Rolloff/rolloff_proto_20191115_2x2_00"
        #impath_90 = "/Volumes/KINGSTON/Villefranche/Prototype/Rolloff/rolloff_proto_20191128_2x2_90_02"
        impath_90 = "/Volumes/KINGSTON/Villefranche/Prototype/Rolloff/rolloff_proto_20191128_2x2_90_01"

    # Path containing the image to analyze
    #impath = processing.folder_choice()

    images_path_00 = glob.glob(impath_00 + "/IMG_*.tif")
    images_path_90 = glob.glob(impath_90 + "/IMG_*.tif")

    images_path_00.sort()
    images_path_90.sort()

    image_path_dark_00 = glob.glob(impath_00 + "/DARK_*.tif")

    # Opening dark image
    imdark_00, metdark_00 = processing.readTIFF_xiMU(image_path_dark_00[0])

    images_path_00 = images_path_00[1:-1]
    #images_path_90 = images_path_90[5:-5]

    #images_path_00 = images_path_00
    images_path_90 = images_path_90[5:-5]

    # Opening geometric calibration results
    dirpath = os.path.dirname(os.getcwd())

    if "4x4" in impath_00 and impath_90:
        geocalib = np.load(dirpath + "/geometric/geometric_calibrationfiles_air/geo_calibration_results_4x4_20191115.npz")
    elif "2x2" in impath_00 and impath_90:
        geocalib = np.load(dirpath + "/geometric/geometric_calibrationfiles_air/geo_calibration_results_2x2_20191115.npz")

    # Zenith and azimuth
    zenith, azimuth = processing.angularcoordinates(geocalib["imagesize"], geocalib["centerpoint"], geocalib["fitparams"])
    cond = zenith > 110
    zenith[cond] = np.nan
    azimuth[cond] = np.nan

    X = np.sin(np.deg2rad(zenith)) * np.sin(np.deg2rad(azimuth))
    Y = np.sin(np.deg2rad(zenith)) * np.cos(np.deg2rad(azimuth))
    Z = np.cos(np.deg2rad(zenith))

    #degFromX = -1 * processing.dwnsampling(np.degrees(np.arctan2(X, Z)), "BGGR")
    degFromX = processing.dwnsampling(np.degrees(np.arctan2(X, Z)), "BGGR")
    degFromY = processing.dwnsampling(np.degrees(np.arctan2(Y, Z)), "BGGR")

    zenith = processing.dwnsampling(zenith, "BGGR")
    azimuth = processing.dwnsampling(azimuth, "BGGR")

    #plt.figure()
    #plt.imshow(zenith[:, :, 1])
    # plt.plot(geocalib["centerpoint"][0]/2, geocalib["centerpoint"][1]/2, "r+")
    # plt.figure()
    # plt.imshow(degFromY_dw[:, :, 0])
    #plt.show()

    # Pre-allocation
    im_total_00 = np.zeros(zenith.shape)
    im_total_90 = np.zeros(zenith.shape)
    im_total_val = np.zeros(zenith.shape)

    # Figure
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()

    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    ax3 = fig3.add_subplot(111)

    fig6, ax6 = plt.subplots(2, 3, figsize=(9, 7))

    axelist = [ax1, ax2, ax3]

    # Looping for 0 deg azimuth
    # Pre-allocation for the loop
    rolloff_00 = np.empty((len(images_path_00), 3))
    rolloff_00.fill(np.nan)

    rolloff_00_centro = np.empty((len(images_path_00), 3))
    rolloff_00_centro.fill(np.nan)

    centroids_00 = np.empty((len(images_path_00), 3), dtype=[("y", "float32"), ("x", "float32")])
    centroids_00.fill(np.nan)

    angles_00 = np.arange(-95, 100, 5)
    angle_pixel_00 = degFromY[int(geocalib["centerpoint"][1]/2), :, :]
    deg_mean_00 = 3  # 3 degrees mean

    for n, path in enumerate(images_path_00):
        print("Processing image number {0}".format(n))
        # Reading data
        im, met = processing.readTIFF_xiMU(path)
        im -= imdark_00
        im_dws = processing.dwnsampling(im, "BGGR", ave=True)

        im_total_00 += im_dws

        for i in range(im_dws.shape[2]):

            # Pixel closer to current angle
            xmin = np.nanargmin(abs(angle_pixel_00[:, i] - (angles_00[n] - deg_mean_00)))
            x = np.nanargmin(abs(angle_pixel_00[:, i] - angles_00[n]))
            xmax = np.nanargmin(abs(angle_pixel_00[:, i] - (angles_00[n] + deg_mean_00)))
            print(xmin, x, xmax)

            nb_pixel_y = 2
            ymin = int(geocalib["centerpoint"][1]/2) - nb_pixel_y
            ymax = int(geocalib["centerpoint"][1]/2) + nb_pixel_y + 1

            im_total_val[ymin:ymax:1, xmin:xmax+1:1, i] = im_dws[ymin:ymax:1, xmin:xmax+1:1, i]

            # Roll - off
            rolloff_00[n, i] = np.mean(im_dws[ymin:ymax:1, xmin:xmax + 1, i])

            # Figures
            axelist[i].plot(x, geocalib["centerpoint"][1] / 2, marker="+", color="r", markersize=4)
            ax6[0, i].plot(x, geocalib["centerpoint"][1] / 2, marker="+", color="r", markersize=4)

            # Centroid
            binary_im, regionprops = processing.regionproperties(im_dws[:, :, i], 0.6E3, 4E3)
            if regionprops:
                centroids_00[n, i] = regionprops[0].centroid
                yc, xc = regionprops[0].centroid
                print(zenith[int(geocalib["centerpoint"][1]/2), int(centroids_00[n, i][1]), i])
                rolloff_00_centro[n, i] = np.mean(im_dws[int(yc), int(xc)-2:int(xc)+3:1, i])

    # Normalization
    print(rolloff_00)
    normrolloff_00 = rolloff_00 / np.nanmax(rolloff_00, axis=0)
    normrolloff_00_centro = rolloff_00_centro / np.nanmax(rolloff_00_centro, axis=0)

    # Looping for 90 deg azimuth
    # Pre-allocation for the loop
    rolloff_90 = np.empty((len(images_path_90), 3))
    rolloff_90.fill(np.nan)

    rolloff_90_centro = np.empty((len(images_path_90), 3))
    rolloff_90_centro.fill(np.nan)

    centroids_90 = np.empty((len(images_path_90), 3), dtype=[("y", "float32"), ("x", "float32")])
    centroids_90.fill(np.nan)

    angles_90 = np.arange(-80, 85, 5)
    angle_pixel_90 = degFromX[:, int(geocalib["centerpoint"][0]/2), :]
    deg_mean_90 = 3  # 5 degrees mean

    for n, path in enumerate(images_path_90):
        print("Processing image number {0}".format(n))
        # Reading data
        im, met = processing.readTIFF_xiMU(path)
        im_dws = processing.dwnsampling(im, "BGGR", ave=True)

        im_total_90 += im_dws

        for i in range(im_dws.shape[2]):
            # Pixel closer to current angle
            #ymin = np.nanargmin(abs(angle_pixel_90[:, i] - (angles_90[n] - deg_mean_90)))
            y = np.nanargmin(abs(angle_pixel_90[:, i] - angles_90[n]))
            #ymax = np.nanargmin(abs(angle_pixel_90[:, i] - (angles_90[n] + deg_mean_90)))
            #print(ymin, y, ymax)

            yvect = np.array([np.nanargmin(abs(angle_pixel_90[:, i] - (angles_90[n] - deg_mean_90))),
                              np.nanargmin(abs(angle_pixel_90[:, i] - (angles_90[n] + deg_mean_90)))])

            ymin = np.min(yvect)
            ymax = np.max(yvect)

            nb_pixel_x = 2
            xmin = int(geocalib["centerpoint"][0]/2) - nb_pixel_x
            xmax = int(geocalib["centerpoint"][0]/2) + nb_pixel_x + 1

            print(ymin, y, ymax)

            # Roll-off
            rolloff_90[n, i] = np.mean(im_dws[ymin:ymax + 1:1, xmin:xmax:1, i])

            im_total_val[ymin:ymax+1:1, xmin:xmax:1, i] = im_dws[ymin:ymax+1:1, xmin:xmax:1, i]

            # Figure
            axelist[i].plot(int(geocalib["centerpoint"][0] / 2), y, marker="+", color="r", markersize=4)
            ax6[1, i].plot(int(geocalib["centerpoint"][0] / 2), y, marker="+", color="r", markersize=4)

            binary_im, regionprops = processing.regionproperties(im_dws[:, :, i], 0.6E3, 4E3)
            if regionprops:
                centroids_90[n, i] = regionprops[0].centroid
                yc, xc = regionprops[0].centroid
                rolloff_90_centro[n, i] = np.mean(im_dws[int(yc)-3:int(yc)+4:1, int(xc), i])

    # Normalization
    print(rolloff_90)
    normrolloff_90 = rolloff_90/np.nanmax(rolloff_90, axis=0)
    normrolloff_90_centro = rolloff_90_centro/np.nanmax(rolloff_90_centro, axis=0)

    print(angles_00)
    print(normrolloff_00)

    # Image
    ax1.imshow(im_total_val[:, :, 0])
    ax1.axhline(geocalib["centerpoint"][1]/2, 0, zenith.shape[1])
    ax1.axvline(geocalib["centerpoint"][0]/2, 0, zenith.shape[0])

    ax2.imshow(im_total_val[:, :, 1])
    ax2.axhline(geocalib["centerpoint"][1] / 2, 0, zenith.shape[1])
    ax2.axvline(geocalib["centerpoint"][0] / 2, 0, zenith.shape[0])

    ax3.imshow(im_total_val[:, :, 2])
    ax3.axhline(geocalib["centerpoint"][1] / 2, 0, zenith.shape[1])
    ax3.axvline(geocalib["centerpoint"][0] / 2, 0, zenith.shape[0])

    # Rolloff figure absolute angle
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)

    ax4.plot(abs(angles_00), normrolloff_00[:, 0], "ro", markersize=4, markerfacecolor="None", markeredgewidth=1, label="0˚ azimuth")
    ax4.plot(abs(angles_00), normrolloff_00[:, 1], "go", markersize=4, markerfacecolor="None", markeredgewidth=1)
    ax4.plot(abs(angles_00), normrolloff_00[:, 2], "bo", markersize=4, markerfacecolor="None", markeredgewidth=1)

    ax4.plot(abs(angles_90), normrolloff_90[:, 0], "rd", markersize=4, markerfacecolor="None", markeredgewidth=1, label="90˚ azimuth")
    ax4.plot(abs(angles_90), normrolloff_90[:, 1], "gd", markersize=4, markerfacecolor="None", markeredgewidth=1)
    ax4.plot(abs(angles_90), normrolloff_90[:, 2], "bd", markersize=4, markerfacecolor="None", markeredgewidth=1)

    ax4.set_xlabel("Angles [degrees]")
    ax4.set_ylabel("Roll-off relative to maximum")

    ax4.legend(loc="best")

    # Rolloff figure
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)

    ax5.plot(angles_00, normrolloff_00[:, 0], "r", label="0˚ azimuth")
    ax5.plot(angles_00, normrolloff_00[:, 1], "g")
    ax5.plot(angles_00, normrolloff_00[:, 2], "b")

    # ax5.plot(normrolloff_00[:, 0], "r")
    # ax5.plot(normrolloff_00[:, 1], "g")
    # ax5.plot(normrolloff_00[:, 2], "b")

    ax5.plot(angles_90, normrolloff_90[:, 0], "r--", label="90˚ azimuth")
    ax5.plot(angles_90, normrolloff_90[:, 1], "g--")
    ax5.plot(angles_90, normrolloff_90[:, 2], "b--")

    ax5.set_xlabel("Angles [degrees]")
    ax5.set_ylabel("Roll-off relative to maximum")

    ax5.legend(loc="best")

    # Figure imtotal
    for n in range(ax6.shape[1]):
        ax6[0, n].imshow(im_total_00[:, :, n])
        ax6[0, n].axhline(geocalib["centerpoint"][1] / 2, 0, zenith.shape[1])
        ax6[0, n].axvline(geocalib["centerpoint"][0] / 2, 0, zenith.shape[0])

        ax6[1, n].imshow(im_total_90[:, :, n])
        ax6[1, n].axhline(geocalib["centerpoint"][1] / 2, 0, zenith.shape[1])
        ax6[1, n].axvline(geocalib["centerpoint"][0] / 2, 0, zenith.shape[0])

    # Figure 8
    fig8 = plt.figure()
    ax8 = fig8.add_subplot(111)

    #ax8.plot(angles_00, normrolloff_00_centro[:, 0], "r")
    #ax8.plot(angles_00, normrolloff_00_centro[:, 1], "g")
    #ax8.plot(angles_00, normrolloff_00_centro[:, 2], "b")

    ax8.plot(angles_90, normrolloff_90_centro[:, 0], "r--")
    ax8.plot(angles_90, normrolloff_90_centro[:, 1], "g--")
    ax8.plot(angles_90, normrolloff_90_centro[:, 2], "b--")

    plt.show()
