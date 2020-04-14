# -*- coding: utf-8 -*-
"""
Python file to performed roll-off experiment data analysis. This roll-off calibration is using experimental setup at
Villefranche-sur-Mer radiometric lab.
"""

# Importation of standard module
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.io import loadmat
import time

# Importation of other modules
import cameracontrol.cameracontrol as cc


# Functions
def rolloff_matlab(x, a1, a2, a3):
    return 1 + a1 * (x ** 2) + a2 * (x ** 4) + a3 * (x ** 6)


if __name__ == "__main__":

    # *** Code beginning ***
    # Creating object from class ProcessImage
    processing = cc.ProcessImage()

    # Input lens to analyzed
    while True:
        answer = input("Which lens do you want to analyze? (c/f): ")
        if answer.lower() in ["c", "f"]:
            break

    di = os.path.dirname(os.getcwd())
    if answer.lower() == "c":
        # Parameter to open the images
        wlens = "close"

        # Lens Close
        impath_00 = "/Volumes/KINGSTON/Villefranche/Insta360/Rolloff/LensClose/20191119_rolloff_00_01"
        impath_90 = "/Volumes/KINGSTON/Villefranche/Insta360/Rolloff/LensClose/20191120_rolloff_90_02"

        # Opening geometric calibration
        geocal = np.load(di + "/geometric/geometric_calibrationfiles_air/geo_LensClose_calibration_results_20191112.npz")

        # Opening fit made previouly with integrating sphere
        fitsphere = loadmat("/Users/raphaellarouche/Desktop/CalibVillefranche/FitRolloff_air_LensClose.mat")
        fitsphere = fitsphere["cvaluesClose"]

        # Central point according to chessboard target calibration
        cpoint = np.array([1731 - 1, 1711 - 1])

        factor_y = -1
        factor_x = -1

    elif answer.lower() == "f":
        # Parameter to open the images
        wlens = "far"

        # Lens Far
        impath_00 = "/Volumes/KINGSTON/Villefranche/Insta360/Rolloff/LensFar/20191119_rolloff_00_01"
        impath_90 = "/Volumes/KINGSTON/Villefranche/Insta360/Rolloff/LensFar/20191119_rolloff_90_01"

        geocal = np.load(di + "/geometric/geometric_calibrationfiles_air/geo_LensFar_calibration_results_20191121.npz")

        # Opening fit made previouly with integrating sphere
        fitsphere = loadmat("/Users/raphaellarouche/Desktop/CalibVillefranche/FitRolloff_air_LensFar.mat")
        fitsphere = fitsphere["cvaluesFar"]

        # Central point according to chessboard target calibration
        cpoint = np.array([1739 - 1, 1716 - 1])

        factor_y = 1
        factor_x = -1

    # Creating a directory with all the files
    image_path_00 = glob.glob(impath_00 + "/IMG_*.dng")
    image_path_90 = glob.glob(impath_90 + "/IMG_*.dng")

    image_path_00.sort()
    image_path_90.sort()

    # Directory path for dark image
    image_path_dark_00 = glob.glob(impath_00 + "/DARK_*.dng")
    image_path_dark_90 = glob.glob(impath_90 + "/DARK_*.dng")

    # Opening dark image
    imdark_00, metdark_00 = processing.readDNG_insta360(image_path_dark_00[0], which_image=wlens)
    imdark_90, metdark_90 = processing.readDNG_insta360(image_path_dark_90[0], which_image=wlens)

    imdark = np.empty((int(metdark_00["Image ImageLength"].values[0] / 2),
                       int(metdark_00["Image ImageWidth"].values[0]), 2), dtype="int64")

    imdark[:, :, 0] = imdark_00
    imdark[:, :, 1] = imdark_90

    # Angles scanned
    angles = np.arange(-90, 95, 5)

    # Zenith and azimuth
    #cpoint = geocal["centerpoint"]

    zenith, azimuth = processing.angularcoordinates(geocal["imagesize"], cpoint, geocal["fitparams"])
    cond = zenith > 90
    zenith[cond] = np.nan
    azimuth[cond] = np.nan

    X = np.sin(np.deg2rad(zenith)) * np.sin(np.deg2rad(azimuth))
    Y = np.sin(np.deg2rad(zenith)) * np.cos(np.deg2rad(azimuth))
    Z = np.cos(np.deg2rad(zenith))

    bayer = "RGGB"
    degFromX = factor_x * processing.dwnsampling(np.degrees(np.arctan2(X, Z)), bayer)
    degFromY = factor_y * processing.dwnsampling(np.degrees(np.arctan2(Y, Z)), bayer)

    zenith = processing.dwnsampling(zenith, bayer)
    azimuth = processing.dwnsampling(azimuth, bayer)

    print(zenith[int(cpoint[1]/2), int(cpoint[0]/2), 0])

    # Show metadata to retrieve the tags
    #processing.show_metadata_insta360(metdark_00)

    # LOOP
    # Pre-allocation of variables
    imtotal = np.zeros((int(metdark_00["Image ImageLength"].values[0]/4), int(metdark_00["Image ImageWidth"].values[0]/2), 3))

    centroids = np.empty((len(image_path_00), 3, 2), dtype=[("y", "float32"), ("x", "float32")])
    centroids.fill(np.nan)
    rolloff = np.empty((len(image_path_00), 3, 2))
    rolloff.fill(np.nan)

    # Pre-allocation of figures
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()

    ax1 = fig1.add_subplot(111)  # Addition of all images red
    ax2 = fig2.add_subplot(111)  # Addition of all images green
    ax3 = fig3.add_subplot(111)  # Addition of all images blue

    list_axe = [ax1, ax2, ax3]

    deltadeg = 5  # 5 degrees around the current point
    # Main loop to for image processing
    for n, path in enumerate(zip(image_path_00, image_path_90)):
        print("Processing image number {0}".format(n))
        for i in range(len(path)):
            # Reading data
            im_op, metadata = processing.readDNG_insta360(path[i], which_image=wlens)
            im_op -= imdark[:, :, i]

            # Downsampling
            im_dws = processing.dwnsampling(im_op, bayer, ave=True)

            for j in range(im_dws.shape[2]):
                _, region_properties = processing.regionproperties(im_dws[:, :, j], 0.9E3, 1E4)

                if region_properties:

                    centroids[n, j, i] = region_properties[0].centroid

                    deglist = degFromY[int(cpoint[1]/2), :, j]

                    y = int(cpoint[1]/2)

                    xvect = np.array([np.nanargmin(abs(deglist - (angles[n] - deltadeg))),
                                      np.nanargmin(abs(deglist - (angles[n] + deltadeg)))])

                    xmin = np.min(xvect)
                    xmax = np.max(xvect)
                    #xmin = np.nanargmin(abs(deglist - (angles[n] - deltadeg)))
                    x = np.nanargmin(abs(deglist - angles[n]))
                    #xmax = np.nanargmin(abs(deglist - (angles[n] + deltadeg)))

                    rolloff[n, j, i] = np.nanmean(im_dws[y, xmin:xmax+1:1, j])

                    if i==1:
                        deglist = degFromX[:, int(cpoint[0]/2), j]

                        x = int(cpoint[0]/2)

                        yvect = np.array([np.nanargmin(abs(deglist - (angles[n] - deltadeg))),
                                          np.nanargmin(abs(deglist - (angles[n] + deltadeg)))])

                        ymin = np.min(yvect)
                        ymax = np.max(yvect)
                        #ymin = np.nanargmin(abs(deglist - (angles[n] - deltadeg)))
                        y = np.nanargmin(abs(deglist - angles[n]))
                        #ymax = np.nanargmin(abs(deglist - (angles[n] + deltadeg)))

                        yvect = np.array([ymin, ymax])

                        rolloff[n, j, i] = np.nanmean(im_dws[ymin:ymax+1:1, x, j])

                    #y, x = region_properties[0].centroid

                    #rolloff[n, j, i] = np.mean(im_dws[ymin:ymax+1:1, xmin:xmax+1:1, j])
                    list_axe[j].plot(x, y, marker="+", color="r", markersize=4)

            # Addition of all image
            imtotal += im_dws

    print(centroids)
    print(rolloff)

    # Normalization rolloff
    normrolloff_00 = rolloff[:, :, 0]/np.nanmax(rolloff[:, :, 0], axis=0)
    normrolloff_90 = rolloff[:, :, 1]/np.nanmax(rolloff[:, :, 1], axis=0)

    # Clip the addition image
    imtotal = np.clip(imtotal, 0, 2**14)

    # Figures
    # Figure of the addition image
    ax1.imshow(imtotal[:, :, 0])
    ax1.axhline(cpoint[1]/2, 0, zenith.shape[1])
    ax1.axvline(cpoint[0]/2, 0, zenith.shape[0])
    ax1.plot(int(geocal["centerpoint"][0]/2), int(geocal["centerpoint"][1]/2), 'ko', markerfacecolor="None", markeredgewidth=1)

    ax2.imshow(imtotal[:, :, 1])
    ax2.axhline(cpoint[1]/2, 0, zenith.shape[1])
    ax2.axvline(cpoint[0]/2, 0, zenith.shape[0])
    ax2.plot(int(geocal["centerpoint"][0]/2), int(geocal["centerpoint"][1] / 2), 'ko', markerfacecolor="None", markeredgewidth=1)

    ax3.imshow(imtotal[:, :, 2])
    ax3.axhline(cpoint[1]/2, 0, zenith.shape[1])
    ax3.axvline(cpoint[0]/2, 0, zenith.shape[0])
    ax3.plot(int(geocal["centerpoint"][0]/2), int(geocal["centerpoint"][1]/2), 'ko', markerfacecolor="None", markeredgewidth=1)

    # Figure of roll-off
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)  # Roll-off

    ax4.plot(angles, normrolloff_00[:, 0], 'r', label="Azimut 0˚")
    ax4.plot(angles, normrolloff_00[:, 1], 'g')
    ax4.plot(angles, normrolloff_00[:, 2], 'b')

    ax4.plot(angles, normrolloff_90[:, 0], 'r--', label="Azimuth 90˚")
    ax4.plot(angles, normrolloff_90[:, 1], 'g--')
    ax4.plot(angles, normrolloff_90[:, 2], 'b--')

    ax4.set_xlabel("Angles [degrees]")
    ax4.set_ylabel("Roll-off relative to maximum")

    ax4.legend(loc="best")

    # Figure of roll-off 2
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)

    ax5.plot(abs(angles), normrolloff_00[:, 0], "ro", markersize=3, markerfacecolor="None", markeredgewidth=1, label="0˚ azimuth")
    ax5.plot(abs(angles), normrolloff_00[:, 1], "go", markersize=3, markerfacecolor="None", markeredgewidth=1)
    ax5.plot(abs(angles), normrolloff_00[:, 2], "bo", markersize=3, markerfacecolor="None", markeredgewidth=1)

    ax5.plot(abs(angles), normrolloff_90[:, 0], "rd", markersize=3, markerfacecolor="None", markeredgewidth=1, label="90˚ azimuth")
    ax5.plot(abs(angles), normrolloff_90[:, 1], "gd", markersize=3, markerfacecolor="None", markeredgewidth=1)
    ax5.plot(abs(angles), normrolloff_90[:, 2], "bd", markersize=3, markerfacecolor="None", markeredgewidth=1)

    ax5.plot(np.linspace(0, 95, 1000), rolloff_matlab(np.linspace(0, 95, 1000), *fitsphere[0, :]), "r-.", label="Fit integrating sphere")
    ax5.plot(np.linspace(0, 95, 1000), rolloff_matlab(np.linspace(0, 95, 1000), *fitsphere[1, :]), "g-.")
    ax5.plot(np.linspace(0, 95, 1000), rolloff_matlab(np.linspace(0, 95, 1000), *fitsphere[2, :]), "b-.")

    ax5.set_xticks(np.arange(0, 100, 10))
    ax5.set_xlim((-5, 95))

    ax5.set_xlabel("Angles [degrees]")
    ax5.set_ylabel("Roll-off relative to maximum")

    ax5.legend(loc="best")

    # Figure 6 - position of center
    im_verifcenter, _ = processing.readDNG_insta360(image_path_00[0], which_image=wlens)

    fig6 = plt.figure()
    ax6 = fig6.add_subplot(111)
    ax6.imshow(im_verifcenter)
    ax6.axhline(cpoint[1], 0, 3456)
    ax6.axvline(cpoint[0], 0, 3456)
    ax6.plot(int(geocal["centerpoint"][0]), int(geocal["centerpoint"][1]), 'ko', markerfacecolor="None", markeredgewidth=1)

    plt.show()
