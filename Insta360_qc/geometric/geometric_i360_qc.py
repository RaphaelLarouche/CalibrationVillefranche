# -*- coding: utf-8 -*-
"""
 Code to read Matlab structure obtained with estimateFisheyeParameters function (based on Scaramuzza calibration toolbox).
"""

__author__ = "Raphael Larouche"

# Importation of standard package
import numpy as np
import inspect
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

import cameracontrol.cameracontrol as cc


# Function
def imageradialdist(imsize, distcenter):
    """
    Function to calculate image radial distance from distortion center.
    :param imsize:
    :param distcenter:
    :return:
    """
    u, v = np.meshgrid(np.arange(1, imsize[1] + 1, 1), np.arange(1, imsize[0] + 1, 1))
    uprim, vprim = u - distcenter[0], v - distcenter[1]

    return np.sqrt(uprim ** 2 + vprim ** 2)


def imagingfunction(MapCoeff, raddistance):
    """
    Imaging function as implemented in Scaramuzza et al.

    :param MapCoeff:
    :param raddistance:
    :return:
    """
    return MapCoeff[0] + MapCoeff[1] * raddistance ** 2 + MapCoeff[2] * raddistance ** 3 + MapCoeff[
        3] * raddistance ** 4


def thetafromprojection(MapCoeff, radial):
    """

    :param imgfunction:
    :param radial:
    :return:
    """
    g = imagingfunction(MapCoeff, radial)

    return np.arctan2(radial, g) * 180 / np.pi


def fitprojectioninv(x, a1, a2, a3):
    return a1 * x + a2 * x ** 2 + a3 * x ** 3


def printfitresults(popt, pcov):
    """

    :param popt:
    :param pcov:
    :return:
    """
    args = []
    pvar = np.diag(pcov)
    for num, x in enumerate(inspect.signature(fitprojectioninv).parameters.items()):
        if num > 0:
            args.append(str(x[0]))
    print("Fit argument values and standard deviation")
    for i, j, k in zip(args, popt, pvar):
        print("{0}: {1:0.3e} +/- {2:0.3e}".format(i, j, k))


def pixeltheta(Intrinsics):
    """

    :param Intrinsics:
    :return:
    """
    MapCoeff = Intrinsics["MappingCoefficients"]
    radial = imageradialdist(Intrinsics["ImageSize"], Intrinsics["DistortionCenter"])

    g_rad = imagingfunction(MapCoeff, radial)
    theta = thetafromprojection(g_rad, radial)

    return theta, radial


def plotreprojectionerrors(params, title, axe):
    # Creation of figure
    ax = axe

    reproErrors = params["ReprojectionErrors"]
    immean = []

    uextre = min(reproErrors[:, 0, :].flatten()), max(reproErrors[:, 0, :].flatten())
    vextre = min(reproErrors[:, 1, :].flatten()), max(reproErrors[:, 1, :].flatten())

    base = 5
    uextre_r = base * round(uextre[0] / base), base * round(uextre[1] / base)
    vextre_r = base * round(vextre[0] / base), base * round(vextre[1] / base)

    val_u = max(map(lambda a: abs(a), uextre_r))
    val_v = max(map(lambda a: abs(a), vextre_r))

    for imnum in range(reproErrors.shape[2]):
        xerr = reproErrors[:, 0, imnum]
        yerr = reproErrors[:, 1, imnum]

        ax.scatter(xerr, yerr, alpha=0.5)

        norm = np.sqrt(xerr ** 2 + yerr ** 2)
        normmean = np.mean(norm)
        immean.append(normmean)
        print("Image {0:d} mean reprojection error: {1:.3f}".format(imnum + 1, normmean))

    text_reprojection_error = "Mean reprojection error: {0:.3f} px\nNumber of images: {1:d}".format(
        sum(immean) / len(immean), len(immean))
    print(text_reprojection_error)

    ax.set_xticks(np.arange(-val_u - 5, val_u + 5, step=5))
    ax.set_yticks(np.arange(-val_v - 5, val_v + 5, step=5))

    ax.set_xlim((-val_u, val_u))
    ax.set_ylim((-val_v, val_v))

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    dy = 0.1 * (2 * val_v)

    ax.text(-val_u + 2, val_v - dy, text_reprojection_error, fontsize=9)
    ax.set_xlabel("Errors u axis [px]")
    ax.set_ylabel("Errors v axis [px]")
    ax.set_title(title)

    return ax


if __name__ == "__main__":

    # **** MAIN CODE START ****

    # Instance of MatlabGeometric
    mgeometric = cc.MatlabGeometric()

    # Importation of data
    filespath = "matlabcalibrationfiles/"

    CloseParams = mgeometric.loadmat(filespath + "CloseFisheyeParams.mat")
    CloseParams = CloseParams["struct_fisheyeParams"]

    CloseErrors = mgeometric.loadmat(filespath + "CloseParamsErrors.mat")
    CloseErrors = CloseErrors["struct_fishError"]

    FarParams = mgeometric.loadmat(filespath + "FarFisheyeParams.mat")
    FarParams = FarParams["struct_fisheyeParams"]

    FarErrors = mgeometric.loadmat(filespath + "FarParamsErrors.mat")
    FarErrors = FarErrors["struct_fishError"]

    print(CloseErrors)
    print(FarErrors)

    # Processing to retrieve theta and rho
    CloseIntrinsics = CloseParams["Intrinsics"]
    FarIntrinsics = FarParams["Intrinsics"]

    CloseTheta, CloseRadial = pixeltheta(CloseIntrinsics)
    FarTheta, FarRadial = pixeltheta(FarIntrinsics)

    # Imaging equation formatting
    CloseEq = "$g_1(r) = ({0:.2E}) ({1:.2E}) \cdot r^2 + ({2:.2E}) \cdot r^3 ({3:.2E}) \cdot r^4$".format(*CloseIntrinsics["MappingCoefficients"])
    FarEq = "$g_2(r) = ({0:.2E}) ({1:.2E}) \cdot r^2 + ({2:.2E}) \cdot r^3 ({3:.2E}) \cdot r^4$".format(*FarIntrinsics["MappingCoefficients"])

    print(CloseEq)
    print(FarEq)

    # Radial vector reduced
    xfit = np.linspace(0, int(np.max(CloseRadial.flatten())), 1000, endpoint=True)
    ReduceCloseTheta = thetafromprojection(CloseIntrinsics["MappingCoefficients"], xfit)
    ReduceFarTheta = thetafromprojection(FarIntrinsics["MappingCoefficients"], xfit)

    # FOV limit
    radlimit = 1600
    CloseFOV = thetafromprojection(CloseIntrinsics["MappingCoefficients"], radlimit)
    FarFOV = thetafromprojection(FarIntrinsics["MappingCoefficients"], radlimit)

    print(r"Lens close FOV: {0:.2f} $\degree$".format(CloseFOV))
    print(r"Lens far FOV: {0:.2f} $\degree$".format(FarFOV))

    # Curve fitting
    # Lens Close
    poptC, pcovC = curve_fit(fitprojectioninv, CloseRadial.flatten(), CloseTheta.flatten())
    pvarC = np.diag(pcovC)
    print(poptC)
    printfitresults(poptC, pcovC)  # Function to print fit arguments

    # Lens Far
    poptF, pcovF = curve_fit(fitprojectioninv, FarRadial.flatten(), FarTheta.flatten())
    pvarF = np.diag(pcovF)
    print(poptF)
    printfitresults(poptF, pcovF)  # Function to print fit arguments

    # Reprojection errors in degrees
    r_function_c, r_corners_c, _, _, _, _ = mgeometric.reprojection_errors(CloseParams)
    r_function_f, r_corners_f, _, _, _, _, = mgeometric.reprojection_errors(FarParams)

    err_deg_c = abs(thetafromprojection(CloseIntrinsics["MappingCoefficients"], r_function_c) - \
                thetafromprojection(CloseIntrinsics["MappingCoefficients"], r_corners_c))

    err_deg_f = abs(thetafromprojection(FarIntrinsics["MappingCoefficients"], r_function_f) - \
                    thetafromprojection(FarIntrinsics["MappingCoefficients"], r_corners_f))

    # Number of images
    Nc = CloseParams["ReprojectionErrors"].shape[2]
    Nf = FarParams["ReprojectionErrors"].shape[2]

    # Figures, visualisation
    plt.style.use("~/PycharmProjects/Insta360_paper_figures/PaperDoubleFig2.mplstyle.txt")
    # Creation of figure
    fig1 = plt.figure()
    fig2 = plt.figure()

    # Creation of axe
    ax1 = fig1.add_subplot(111)
    ax2_a = fig2.add_subplot(221)
    ax2_proj1 = fig2.add_subplot(222)
    ax2_proj2 = fig2.add_subplot(224)

    # Fontsize and markersize
    #plt.rcParams.update({'font.size': 13})  # Default fontsize
    #legendfontsize = 12

    # Colorwheel
    colourWheel = ['#329932',
                   '#ff6961',
                   'b',
                   '#6a3d9a',
                   '#fb9a99',
                   '#e31a1c',
                   '#fdbf6f',
                   '#ff7f00',
                   '#cab2d6',
                   '#6a3d9a',
                   '#ffff99',
                   '#b15928',
                   '#67001f',
                   '#b2182b',
                   '#d6604d',
                   '#f4a582',
                   '#fddbc7',
                   '#f7f7f7',
                   '#d1e5f0',
                   '#92c5de',
                   '#4393c3',
                   '#2166ac',
                   '#053061']

    ax1.plot(xfit, ReduceCloseTheta, linestyle="-", color="#1C366A",
             label="First optic")
    ax1.plot(xfit, ReduceFarTheta, linestyle="-.", alpha=0.7, color="#727272",
             label="Second optic")
    ax1.axvline(radlimit, ls="--", linewidth=2, color="#D45B40")

    # ax1.text(100, 65, CloseEq, fontsize=9)
    # ax1.text(100, 60, FarEq, fontsize=9)

    ax1.set_xlim((0, 2000))
    ax1.set_ylim((0, 100))

    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())

    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()

    ax1.tick_params(axis='x', which='minor', top=True, direction="in")
    ax1.tick_params(axis='y', which='minor', left=True, direction="in")
    ax1.tick_params(axis='x', which='major', top=True, direction="in")
    ax1.tick_params(axis='y', which='major', left=True, direction="in")

    ax1.set_xlabel("Radial position [px]")
    ax1.set_ylabel(r"Angle $\theta$ from optic axis [$\degree$]")

    ax1.annotate(r"Image radius limit: {0:.2f}$\degree$".format(CloseFOV),
                 xy=(1598, 75), xycoords='axes fraction', xytext=(0.37, 0.25),
                 textcoords='axes fraction', fontsize=11, fontname="DejaVu Sans", fontweight="bold", color="#D45B40")

    #ax1.annotate('Image radius limit', xy=(1598, 75),  xycoords='data', xytext=(0.5, 0.4), textcoords='axes fraction',
    #             arrowprops=dict(arrowstyle="->", facecolor='black'), fontsize=13)

    ax1.legend(loc="best")

    # ********* MULTIPLE PLOTS *********

    # Distortion function
    ax2_a.plot(xfit, ReduceCloseTheta, linestyle="-", color="#1C366A",
             label="First optic")
    ax2_a.plot(xfit, ReduceFarTheta, linestyle="-.", alpha=0.7, color="#727272",
             label="Second optic")
    ax2_a.axvline(radlimit, ls="--", linewidth=2, color="#D45B40")

    ax2_a.set_xlim((0, 2000))
    ax2_a.set_ylim((0, 100))

    ax2_a.xaxis.set_minor_locator(AutoMinorLocator())
    ax2_a.yaxis.set_minor_locator(AutoMinorLocator())

    ax2_a.yaxis.set_label_position("right")
    ax2_a.yaxis.tick_right()

    ax2_a.tick_params(axis='x', which='minor', top=True, direction="in")
    ax2_a.tick_params(axis='y', which='minor', left=True, direction="in")
    ax2_a.tick_params(axis='x', which='major', top=True, direction="in")
    ax2_a.tick_params(axis='y', which='major', left=True, direction="in")

    ax2_a.set_xlabel("Radial position [px]")
    ax2_a.set_ylabel(r"Angle $\theta$ from optic axis [$\degree$]")

    ax2_a.annotate(r"Image radius limit: {0:.2f}$\degree$".format(CloseFOV),
                 xy=(1598, 75), xycoords='axes fraction', xytext=(0.37, 0.25),
                 textcoords='axes fraction', fontsize=11, fontname="DejaVu Sans", fontweight="bold", color="#D45B40")

    ax2_a.legend(loc="best")

    # Reprojection errors plot
    plotreprojectionerrors(CloseParams, "First optic reprojection errors", ax2_proj1)
    plotreprojectionerrors(FarParams, "Second optic reprojection errors", ax2_proj2)

    # Reprojection error in degrees
    fig3 = plt.figure(figsize=(11, 5))
    ax3_1 = fig3.add_subplot(121)
    ax3_2 = fig3.add_subplot(122)

    ax3_1.plot(xfit, ReduceCloseTheta, linewidth=1.5, linestyle="-", color="black", label="Lens 1")
    ax3_1.plot(xfit, ReduceFarTheta, linewidth=1.5, linestyle="-.", color="gray", label="Lens 2")
    ax3_1.plot(radlimit, CloseFOV, "d", color="#D45B40", markerfacecolor="none", markersize=5)
    #ax3_1.axvline(radlimit, ls="--", linewidth=2, color="#D45B40")

    ax3_1.set_ylim((0, 80))
    ax3_1.set_xlim((0, 1750))
    ax3_1.set_xticks(np.arange(0, 2000, 250))
    ax3_1.set_xlim((0, 1750))

    ax3_1.annotate(r"Image limit: {0:.2f}$\degree$".format(CloseFOV), xy=(1598, 75), xytext=(-150, 0), textcoords='offset points', fontsize=10, color="black", arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"))

    ax3_1.set_xlabel(r"Radial distance $r$ [px]")
    ax3_1.set_ylabel(r"Scene angle $\theta$ [$\degree$]")

    ax3_1.legend(loc="best", fontsize=11)

    text_reprojerr1 = "Lens 1 residuals median: {0:.3f}˚\nLens 2 residuals median: {1:.3f}˚".format(np.median(err_deg_c), np.median(err_deg_f))
    ax3_2.scatter(r_function_c.ravel(), err_deg_c.ravel(), marker="o", s=8, edgecolor="black", facecolor="none", label="Lens 1, {0} acquisitions".format(Nc))
    ax3_2.scatter(r_function_f.ravel(), err_deg_f.ravel(), marker="v", s=8, edgecolor="gray", facecolor="none", label="Lens 2, {0} acquisitions".format(Nf))

    ax3_2.text(10, 2, text_reprojerr1, fontsize=10)

    ax3_2.set_yscale("log")
    ax3_2.set_ylim((0.00001, 10))
    ax3_2.set_xticks(np.arange(0, 1750, 250))
    ax3_2.set_xlim((-105, 1500))

    ax3_2.set_xlabel(r"Radial distance $r$ [px]")
    ax3_2.set_ylabel(r"$\theta$ residuals [˚]")

    ax3_2.legend(loc=4, fontsize=11)

    # Tight layout
    plt.tight_layout()

    # Saving figures
    fig3.savefig("/Users/raphaellarouche/Desktop/Figures_pres20avril/geo_com.eps", dpi=600)

    plt.show()
