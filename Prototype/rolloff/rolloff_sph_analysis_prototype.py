# -*- coding: utf-8 -*-
"""
Python file for roll-off data analysis for the prototype with 2x2 binning (integrating sphere).
"""

# Importation of standard modules
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Importation of other modules
import cameracontrol.cameracontrol as cc


# Functions
def roff_fitcurve(x, a0, a2, a4, a6, a8):
    """
    Roll-off fitting function.

    :param x: radial position from optical center
    :param a0:
    :param a2:
    :param a4:
    :param a6:
    :param a8:
    :return:
    """
    return a0 + a2 * x ** 2 + a4 * x ** 4 + a6 * x ** 6 + a8 * x ** 8


def disp_roll_fitcurve(popt, perr):
    """
    Function to print roll-off fitting results.

    :param popt:
    :param perr:
    :return:
    """
    param = ["a0", "a2", "a4", "a6", "a8"]
    a = ""
    for i in zip(param, popt, perr):
        a += "%s: %.4E (%.4E)\n" % i

    print(a)


def processing_img_list(folderpath, n_remove, axe):

    # Process Image class
    processing = cc.ProcessImage()

    # Angles
    angles = np.arange(-105, 110, 5)
    angles = angles[n_remove:-n_remove]

    # List of image path
    image_path = glob.glob(folderpath + "/IMG_*.tif")
    image_path.sort()
    image_path = image_path[n_remove:-n_remove]

    # Dark image
    image_path_dark = glob.glob(folderpath + "/DARK_*.tif")
    image_path_dark.sort()
    imdark, metdark = processing.readTIFF_xiMU(image_path_dark[0])

    # Pre-allocation
    imtotal = np.zeros((imdark.shape[0] // 2, imdark.shape[1] // 2, 3))

    rolloff = np.empty((len(image_path), 3))
    rolloff.fill(np.nan)

    std_rolloff = np.empty((len(image_path), 3))
    std_rolloff.fill(np.nan)

    centroids = np.empty((len(image_path), 3), dtype=[("y", "float32"), ("x", "float32")])
    centroids.fill(np.nan)

    for n, path in enumerate(image_path):

        print("Processing image number {0}".format(n))

        # Reading data
        im, met = processing.readTIFF_xiMU(path)
        im -= imdark

        print(met["exposure_time_us"])

        im_dws = processing.dwnsampling(im, "BGGR", ave=True)

        imtotal += im_dws

        for i in range(im_dws.shape[2]):

            # Centroids
            bin_im, regprops = processing.regionproperties(im_dws[:, :, i], 0.3E3)

            if regprops:
                centroids[n, i] = regprops[0].centroid
                yc, xc = regprops[0].centroid
                yc = int(round(yc))
                xc = int(round(xc))

                axe[i].plot(xc, yc, "r+")

                ROI = im_dws[yc-2:yc+3:1, xc-2:xc+3:1, i]

                rolloff[n, i] = np.mean(ROI)
                std_rolloff[n, i] = np.std(ROI)

                print(ROI.shape)

    imtotal = np.clip(imtotal, 0, 2 ** 12)
    for n, a in enumerate(axe):
        a.imshow(imtotal[:, :, n])

    return rolloff, std_rolloff, angles, centroids


def plot_rolloff_abs(axe, angle, rolloff, mark, cl, lab=""):
    if lab:
        axe.plot(abs(angle), rolloff, marker=mark, markersize=2, color=cl, markerfacecolor="None", linestyle="None",
                 alpha=0.5, label=lab)
    else:
        axe.plot(abs(angle), rolloff, marker=mark, markersize=2, color=cl, markerfacecolor="None", linestyle="None",
                 alpha=0.5)



if __name__ == "__main__":

    # *** Code beginning ***
    processing = cc.ProcessImage()

    # Image directory creation
    #path_00 = "rolloff"
    #path_00 = "/Volumes/KINGSTON/Quebec/Prototype/Rolloff/rolloff_proto_air/rolloff_proto_20191213_2x2"
    #path_00 = "/Volumes/KINGSTON/Quebec/Prototype/Rolloff/rolloff_proto_air/rolloff_proto_20200117_2x2_02"
    #path_00 = "/Volumes/KINGSTON/Quebec/Prototype/Rolloff/rolloff_proto_air/rolloff_proto_20200225_2x2_02"
    #path_00 = "/Volumes/KINGSTON/Quebec/Prototype/Rolloff/rolloff_proto_air/rolloff_proto_20200225_2x2_90_01"
    #path_00 = "/Volumes/KINGSTON/Quebec/Prototype/Rolloff/rolloff_proto_air/rolloff_proto_20200305_2x2_00_02"

    path_00 = "/Volumes/KINGSTON/Quebec/Prototype/Rolloff/rolloff_proto_air/rolloff_proto_20200306_2x2_00_02"
    path_45 = "/Volumes/KINGSTON/Quebec/Prototype/Rolloff/rolloff_proto_air/rolloff_proto_20200311/rolloff_proto_2020311_2x2_45"
    path_90 = "/Volumes/KINGSTON/Quebec/Prototype/Rolloff/rolloff_proto_air/rolloff_proto_20200225_2x2_90_01"
    path_135 = "/Volumes/KINGSTON/Quebec/Prototype/Rolloff/rolloff_proto_air/rolloff_proto_20200311/rolloff_proto_20200311_2x2_135"

    # ___________________________________________________________________________
    # Getting all roll-off curves
    # Figure roll-off
    fig1, ax1 = plt.subplots(1, 3)

    # Roll-off 0 degree azimuth
    rolloff_00, std_rolloff_00, angles, centroids_00 = processing_img_list(path_00, 1, ax1)
    rolloff_45, std_rolloff_45, angles_45, centroids_45 = processing_img_list(path_45, 5, ax1)
    #rolloff_90, std_rolloff_90, angles_90, centroids_90 = processing_img_list(path_90, 1, ax1)
    rolloff_135, std_rolloff_135, angles_135, centroids_135 = processing_img_list(path_135, 3, ax1)

    # Relative standard error
    RSE = std_rolloff_00 / rolloff_00[None, :]
    RSE = np.squeeze(RSE)

    print(RSE)

    # Normalization of roll-off
    rolloff_00 = rolloff_00 / np.nanmax(rolloff_00, axis=0)
    rolloff_45 = rolloff_45 / np.nanmax(rolloff_45, axis=0)
    #rolloff_90 = rolloff_90 / np.nanmax(rolloff_90, axis=0)
    rolloff_135 = rolloff_135 / np.nanmax(rolloff_135, axis=0)

    # Sorting roll-off
    ind_sroff = np.tile(np.argsort(abs(angles)), (3, 1)).T

    print(ind_sroff[:, 0])

    sangle = np.sort(abs(angles))
    sroff = np.take_along_axis(rolloff_00, ind_sroff, axis=0)
    sRSE = np.take_along_axis(RSE,  ind_sroff, axis=0)

    # ___________________________________________________________________________
    # Fit
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)

    xdata = np.linspace(0, 105, 1000)
    col = ["r", "g", "b"]
    bandnames = {"r": "red channel", "g": "green channel", "b": "blue channel"}
    rolloff_fitparams = {}

    for n in range(rolloff_00.shape[1]):
        val = rolloff_00[:, n]
        ang = abs(angles)

        mask = ~np.isnan(val)

        val = val[mask]
        ang = ang[mask]

        popt, pcov = processing.rolloff_curvefit(ang, val)
        rsquared, perr = processing.rsquare(processing.rolloff_polynomial, popt, pcov, ang, val)

        print(bandnames[col[n]])
        disp_roll_fitcurve(popt, perr)
        print(rsquared)

        # Saving fit params
        rolloff_fitparams[bandnames[col[n]]] = popt

        ax3.plot(xdata, processing.rolloff_polynomial(xdata, *popt), color=col[n], alpha=0.7,
                 label="Polynomial fit {0} ($R^2$={1:.3f})".format(bandnames[col[n]], rsquared))

    print(rolloff_fitparams.values())

    # ___________________________________________________________________________
    # Figure configuration

    # Figure 2 - Shape of the roll-off curves with relative angle
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    ax2.errorbar(angles, rolloff_00[:, 0], RSE[:, 0], alpha=0.5, color="r", label="Red channel")
    ax2.errorbar(angles, rolloff_00[:, 1], RSE[:, 1], alpha=0.5, color="g", label="Green channel")
    ax2.errorbar(angles, rolloff_00[:, 2], RSE[:, 2], alpha=0.5, color="b", label="Blue channel")

    ax2.set_xlabel("Angles [˚]")
    ax2.set_ylabel("Roll-off relative to maximum")

    ax2.legend(loc="best")

    # Figure 3 - Roll-off with absolute angle

    ax3.plot(abs(angles), rolloff_00[:, 0], "ro", alpha=0.5, markersize=4, markerfacecolor="None", markeredgewidth=1)
    ax3.plot(abs(angles), rolloff_00[:, 1], "go", alpha=0.5, markersize=4, markerfacecolor="None", markeredgewidth=1)
    ax3.plot(abs(angles), rolloff_00[:, 2], "bo", alpha=0.5, markersize=4, markerfacecolor="None", markeredgewidth=1)

    ax3.errorbar(abs(angles), rolloff_00[:, 0], RSE[:, 0], linestyle="None", alpha=0.5, color="r")
    ax3.errorbar(abs(angles), rolloff_00[:, 1], RSE[:, 1], linestyle="None", alpha=0.5, color="g")
    ax3.errorbar(abs(angles), rolloff_00[:, 2], RSE[:, 2], linestyle="None", alpha=0.5, color="b")

    ax3.set_xlabel("Angles from optical axis [˚]")
    ax3.set_ylabel("Roll-off")

    ax3.legend(loc="best")

    # Figure 4 - Roll-off for all azimuth angle
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    cllist = ["r", "g", "b"]

    for i in range(3):
        if i == 0:
            plot_rolloff_abs(ax4, abs(angles), rolloff_00[:, i], "o", cllist[i], lab="0˚ azimuth")
            plot_rolloff_abs(ax4, abs(angles_45), rolloff_45[:, i], "s", cllist[i], lab="45˚ azimuth")
            plot_rolloff_abs(ax4, abs(angles_135), rolloff_135[:, i], "<", cllist[i], lab="135˚ azimuth")

        else:
            plot_rolloff_abs(ax4, abs(angles), rolloff_00[:, i], "o", cllist[i])
            plot_rolloff_abs(ax4, abs(angles_45), rolloff_45[:, i], "s", cllist[i])
            plot_rolloff_abs(ax4, abs(angles_135), rolloff_135[:, i], "<", cllist[i])

    ax4.set_xlabel("Angles from optical axis [˚]")
    ax4.set_ylabel("Roll-off")
    ax4.legend(loc="best")

    # ___________________________________________________________________________
    # Saving calibration data
    while True:
        inputsav = input("Do you want to save the calibration results? (y/n) : ")
        inputsav = inputsav.lower()
        if inputsav in ["y", "n"]:
            break

    if inputsav == "y":
        # Air
        name = "rolloff_proto_sph_air_2x2_" + path_00[-15:-7]
        savename = "calibrationfiles/" + name + ".npz"

        np.savez(savename, rolloff_fitparams=rolloff_fitparams)

    plt.show()
