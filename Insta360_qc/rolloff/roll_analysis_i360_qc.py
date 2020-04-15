# -*- coding: utf-8 -*-
"""
Script to reproduce the roll-off curves obtained by analyzing data of sphere output (inside water) for the commercial
camera insta360 ONE. (trying to removes demosaic that was made before)
"""

# Importation of standard modules
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import glob

# Importation of other modules
import cameracontrol.cameracontrol as cc

# Functions

def imageslist(path):

    imlist = glob.glob(path + "/IMG_*.dng")
    imlist.sort()

    return imlist

def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def imagingfunction(MapC, rdistance):
    """
    Imaging function as implemented in Scaramuzza et al.

    :param MapCoeff:
    :param raddistance:
    :return:
    """
    return MapC[0] + MapC[1] * rdistance ** 2 + MapC[2] * rdistance ** 3 + MapC[3] * rdistance ** 4


def pixel_angular_coord(intrinsics):

    imsize = intrinsics["ImageSize"]
    optical_center = intrinsics["DistortionCenter"]

    x, y = np.meshgrid(np.arange(1, imsize[1] + 1, 1), np.arange(1, imsize[0] + 1, 1))
    xprim, yprim = x - optical_center[0], y - optical_center[1]
    rdistance = np.sqrt(xprim ** 2 + yprim ** 2)

    g = imagingfunction(intrinsics["MappingCoefficients"], rdistance)

    az = np.arctan2(yprim, xprim)
    az[az < 0] = az[az < 0] + 2 * np.pi

    return np.arctan2(rdistance, g) * 180 / np.pi, az * 180 / np.pi


def coord3d(zenith, azimuth):
    """
    3D coordinates from zenith and azimuth matrix. These last matrix should be downsampled for each red, green and blue
    pixels.

    :param zenith:
    :param azimuth:
    :return:
    """
    if (len(zenith.shape) == 3) and (len(azimuth.shape) == 3):
        x = np.sin(np.deg2rad(zenith)) * np.cos(np.deg2rad(azimuth))
        y = np.sin(np.deg2rad(zenith)) * np.sin(np.deg2rad(azimuth))
        z = np.cos(np.deg2rad(zenith))
    else:
        raise ValueError("Downsampled before!")

    return x, y, z


def anglefromXaxis(zenith, azimuth):
    """

    :param zenith:
    :param azimuth:
    :return:
    """

    x, y, z = coord3d(zenith, azimuth)

    return np.degrees(np.arctan2(x, z))

def anglefromYaxis(zenith, azimuth):
    """

    :param zenith:
    :param azimuth:
    :return:
    """
    x, y, z = coord3d(zenith, azimuth)

    return np.degrees(np.arctan2(y, z))

def rolloff_loop_i360(imlist, which_lens, Intr, azimuth="0"):
    """

    :param imlist:
    :param which_lens:
    :param Intr:
    :param azimuth:
    :return:
    """
    # Instance of class ProcessImage
    processIm = cc.ProcessImage()

    # Angular coordinates
    zen, az = pixel_angular_coord(Intr)

    zen_dwsa = processIm.dwnsampling(zen, "RGGB", ave=True)
    az_dwsa = processIm.dwnsampling(az, "RGGB", ave=True)

    if azimuth == "0":
        aX = anglefromXaxis(zen_dwsa, az_dwsa)
    elif azimuth == "90":
        aX = anglefromYaxis(zen_dwsa, az_dwsa)
    else:
        raise ValueError("Not an option for azimuth.")

    # Pre-allocation
    imtotal = np.zeros((Intr["ImageSize"][0] // 2, Intr["ImageSize"][1] // 2, 3))
    centroid = np.empty((len(imlist), 3), dtype=[("y", "float32"), ("x", "float32")])
    rolloff = np.empty((len(imlist), 3), dtype=[("a", "float32"), ("a_rel", "float32"), ("DN_avg", "float32"), ("DN_std", "float32")])
    centroid.fill(np.nan)
    rolloff.fill(np.nan)

    for n, path in enumerate(imlist):
        print("Processing image number {0}".format(n))

        # Reading data
        im_op, metadata = processIm.readDNG_insta360(path, which_image=which_lens)
        im_op = im_op.astype(float)

        # Read noise removal
        im_op -= float(str(metadata["Image Tag 0xC61A"]))

        # Downsampling
        im_dws = processIm.dwnsampling(im_op, "RGGB", ave=True)

        # Image total
        imtotal += im_dws

        # Region properties for the centroids
        _, region_properties = processIm.regionproperties(im_dws[:, :, 0], 1E3, 1E4)  # Red image

        # Filtering regionproperties according to the centroid position
        if azimuth == "0":
            region_properties = [reg for reg in region_properties if reg.centroid[0] > Intr["DistortionCenter"][1] // 2 - 30]
        elif azimuth == "90":
            region_properties = [reg for reg in region_properties if
                  (Intr["DistortionCenter"][0] // 2 + 200) > reg.centroid[1] > (Intr["DistortionCenter"][0] // 2 - 200)]

        if region_properties:
            for j in range(im_dws.shape[2]):
                yc, xc = region_properties[0].centroid

                # # y on a line rotated of few degrees to fit the spot
                # dy = np.sin(np.deg2rad(2.3)) * (xc - Intrinsics["DistortionCenter"][0]//2)
                # yc = (Intrinsics["DistortionCenter"][1]//2 + 15) - dy

                # Using constant number of pixel around centroid
                _, data = processIm.data_around_centroid(im_dws[:, :, j], (yc, xc), 15)

                centroid[n, j] = yc, xc  # storing centroid
                rolloff[n, j] = zen_dwsa[int(round(yc)), int(round(xc)), j], \
                                aX[int(round(yc)), int(round(xc)), j], \
                                np.mean(data), \
                                np.std(data)

    return imtotal, rolloff, centroid


def plot_rolloff_1(axe, angles, rolloff, err, mark, cl, lab=""):
    """

    :param axe:
    :param angle:
    :param rolloff:
    :param mark:
    :param cl:
    :param lab:
    :return:
    """

    if lab:
        axe.errorbar(angles, rolloff,
                     yerr=err, ecolor=cl, marker=mark, linestyle="", markerfacecolor=cl,
                     markeredgecolor="black",
                     markersize=4, label=lab)
    else:
        axe.errorbar(angles, rolloff,
                     yerr=err, ecolor=cl, marker=mark, linestyle="", markerfacecolor=cl,
                     markeredgecolor="black",
                     markersize=4, label=lab)

    axe.set_ylabel("Roll-off")


if __name__ == "__main__":

    # Input lens to analyzed
    while True:
        answer = input("Which lens do you want to analyze? (c/f): ")
        if answer.lower() in ["c", "f"]:
            break

    # Instance of ProcessImage
    process = cc.ProcessImage()

    # Open files
    generalpath = "/Volumes/KINGSTON/Insta360/RolloffCharacterization/"  # General path toward images
    generalpathgeo = "/Users/raphaellarouche/PycharmProjects/Insta360_paper_figures/files/geo_calib/"

    if answer.lower() == "c":

        # Roll-off images
        imlist_00 = imageslist(generalpath + "Characterization_05012019_Water/LensClose_2")
        imlist_90 = imageslist(generalpath + "Characterization_07262019_Water/orthogonal_lensclose_01")
        imlist_90 = imlist_90[1:-1]

        # Geometric calibration
        FishParams = loadmat(generalpathgeo + "CloseFisheyeParams.mat")
        FishParams = FishParams["struct_fisheyeParams"]

        # AIR roll-off
        fitmatlab = loadmat("/Users/raphaellarouche/Desktop/CalibVillefranche/FitRolloff_air_LensClose.mat")
        fitmatlab = fitmatlab["cvaluesClose"]
    else:

        # Roll-off images
        imlist_00 = imageslist(generalpath + "Characterization_05032019_Water/LensFar")
        imlist_90 = imageslist(generalpath + "Characterization_07262019_Water/orthogonal_lensfar_01")
        imlist_90 = imlist_90[1:-1]

        # Geometric calibration
        FishParams = loadmat(generalpathgeo + "FarFisheyeParams.mat")
        FishParams = FishParams["struct_fisheyeParams"]

        # Air roll-off
        fitmatlab = loadmat("/Users/raphaellarouche/Desktop/CalibVillefranche/FitRolloff_air_LensFar.mat")
        fitmatlab = fitmatlab["cvaluesFar"]

    # Dictionary to open DNG files using readDNG_insta360
    wlens = {"c": "close", "f": "far"}

    # Angular coordinates of image
    Intrinsics = FishParams["Intrinsics"]
    zen, az = pixel_angular_coord(Intrinsics)
    zen_dws = process.dwnsampling(zen, "RGGB")  # Downsampled
    az_dws = process.dwnsampling(az, "RGGB")  # Downsampled
    angleX = anglefromXaxis(zen_dws, az_dws)

    # - fig
    fig2, ax2 = plt.subplots(1, 1)
    ax2.axhline(y=Intrinsics["DistortionCenter"][1]//2, xmin=0, xmax=3455//2)
    ax2.axvline(x=Intrinsics["DistortionCenter"][0]//2, ymin=0, ymax=3455//2)

    # Loop
    imtotal, roff_centro, centro = rolloff_loop_i360(imlist_00, wlens[answer.lower()], Intrinsics, azimuth="0")
    imtotal_90, roff_centro_90, centro_90 = rolloff_loop_i360(imlist_90, wlens[answer.lower()], Intrinsics, azimuth="90")

    # Clipping image
    imtotal = np.clip(imtotal + imtotal_90, 0, 2**14 - 1)

    # Normalization
    roff_centro_norm = roff_centro.copy()
    roff_centro_norm["DN_avg"] /= np.max(roff_centro["DN_avg"], axis=0)
    roff_centro_norm["DN_std"] /= np.max(roff_centro["DN_avg"], axis=0)

    roff_centro_90_norm = roff_centro_90.copy()
    roff_centro_90_norm["DN_avg"] /= np.max(roff_centro_90["DN_avg"], axis=0)
    roff_centro_90_norm["DN_std"] /= np.max(roff_centro_90["DN_avg"], axis=0)

    print(np.mean(roff_centro["DN_std"], axis=0))

    # Figures
    # Fig1 - zenith, azimuth
    fig1, ax1 = plt.subplots(1, 2)

    ax1[0].imshow(zen)
    ax1[1].imshow(az)

    # Fig2 - image total
    ax2.imshow(imtotal[:, :, 0])
    ax2.plot(centro["x"][:, 0], centro["y"][:, 0], "r+")
    ax2.plot(centro_90["x"][:, 0], centro_90["y"][:, 0], "r+")

    # Fig3 - rolloff using data around centroid
    # fig3 = plt.figure()
    # ax3 = fig3.add_subplot(111)
    fig3, ax3 = plt.subplots(3, 1, sharex=True, figsize=(6.4, 7))

    # Fig5 - rolloff water vs air
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)

    # Fig6 - ratio air vs water
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(111)

    # Fitting
    theta = np.linspace(0, 80, 100)
    marker = ["o", "s", "d"]

    color = iter(['#d62728', '#2ca02c', '#1f77b4'])
    lab = ["red pixels", "green pixels", "blue pixels"]

    for band in range(roff_centro.shape[1]):
        # Fit
        print("Matlab fit")
        popt, pcov, rsquared, perr = process.rolloff_curvefit_matlab(roff_centro_norm["a"][:, band],
                                                                     roff_centro_norm["DN_avg"][:, band])

        print("8 degree fit")
        popt2, pcov2, rsquared2, perr2 = process.rolloff_curvefit(roff_centro_norm["a"][:, band],
                                                                  roff_centro_norm["DN_avg"][:, band])

        popt90, pcov90, rsquared90, perr90 = process.rolloff_curvefit(roff_centro_90_norm["a"][:, band],
                                                                      roff_centro_90_norm["DN_avg"][:, band])

        atot = np.append(roff_centro_norm["a"][:, band], roff_centro_90_norm["a"][:, band])
        rtot = np.append(roff_centro_norm["DN_avg"][:, band], roff_centro_90_norm["DN_avg"][:, band])
        poptall, pcovall, rsquareall, perrall = process.rolloff_curvefit(atot, rtot)

        # Plots
        col = next(color)

        plot_rolloff_1(ax3[band], roff_centro_norm["a"][:, band],
                       roff_centro_norm["DN_avg"][:, band], roff_centro_norm["DN_std"][:, band], "o", col,
                       "0˚ azimuth "+lab[band])

        plot_rolloff_1(ax3[band], roff_centro_90_norm["a"][:, band],
                       roff_centro_90_norm["DN_avg"][:, band], roff_centro_90_norm["DN_std"][:, band], "s", col,
                       "90˚ azimuth "+lab[band])

        #ax3[band].plot(theta, process.rolloff_polynomial(theta, *popt2), color=col, linewidth=1.7, label="Fit 0˚ azimuth")
        #ax3[band].plot(theta, process.rolloff_polynomial(theta, *popt90), color=col, linewidth=1.7, linestyle="-.", label="Fit 90˚ azimuth")
        ax3[band].plot(theta, process.rolloff_polynomial(theta, *poptall), color=col, linewidth=1.7, linestyle="-", label="Polynomial fit")
        ax3[band].text(52, 0.91, "$k=1$ standard uncertainty\n$r^{{2}}={0:.5f}$".format(rsquareall), fontsize=9)

        # Air vs water
        ax5.plot(theta, process.rolloff_matlab(theta, *popt), color=col, linewidth=1.7)
        ax5.plot(theta, process.rolloff_matlab(theta, *fitmatlab[band, :]), color=col, linestyle="-.", linewidth=1.7)

        # Aire/water
        ax6.plot(theta, process.rolloff_matlab(theta, *fitmatlab[band, :])/process.rolloff_matlab(theta, *popt))

    # Fig3
    ax3[0].set_xlim((0, 80))
    ax3[1].set_xlim((0, 80))
    ax3[2].set_xlabel(r"Angle $\theta$ [˚]")

    ax3[0].legend(loc="best")
    ax3[1].legend(loc="best")
    ax3[2].legend(loc="best")

    # Fig4 - rolloff relative
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)

    ax4.plot(roff_centro_norm["a_rel"], roff_centro_norm["DN_avg"], marker="o", linestyle="", markeredgecolor="black")

    ax4.set_xlabel(r"Relative angle [˚]")
    ax4.set_ylabel("Roll-off")

    # Fig5
    ax5.set_xlabel(r"Angle $\theta$ [˚]")
    ax5.set_ylabel("Roll-off")
    ax5.legend(loc="best")

    # Fig6
    fig7 = plt.figure()
    ax7 = fig7.add_subplot(111)

    ax7.plot(roff_centro_90["a"], roff_centro_90["DN_avg"]/np.max(roff_centro_90["DN_avg"], axis=0), "o")

    plt.show()
