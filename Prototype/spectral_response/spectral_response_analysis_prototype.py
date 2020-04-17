# -*- coding: utf-8 -*-

"""
Python script to analyze results from experiment of spectral response of xiMU MT9P031
sensor using spectrophotometer.

"""

#  Importation of standard modules

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.signal import chirp, find_peaks, peak_widths

# Importation of other modules
import cameracontrol.cameracontrol as cc

# Function

if __name__ == "__main__":
    # ___________________________________________________________________________
    # *** Code beginning ***

    # Creating object from class ProcessImage
    processing = cc.ProcessImage()

    # Image path
    path = "/Volumes/KINGSTON/Quebec/Prototype/Spectral_response/spectral_response_2x2_20191206"
    imlist = glob.glob(path + "/IMG_*tif")
    imlist.sort()

    # Spectral response of manufacturer
    genpath = "~/PycharmProjects/CalibrationVillefranche/"
    sensor_rsr_data = pandas.read_csv(genpath + "cameracontrol/MT9P031_RSR/MT9P031.csv", sep=";")
    sensor_rsr_data["R"] = sensor_rsr_data["R"] / np.nanmax(sensor_rsr_data["R"])
    sensor_rsr_data["G"] = sensor_rsr_data["G"] / np.nanmax(sensor_rsr_data["G"])
    sensor_rsr_data["B"] = sensor_rsr_data["B"] / np.nanmax(sensor_rsr_data["B"])

    sensor_rsr_data = sensor_rsr_data.dropna()

    # Incident power with gentec powermeter
    powerdata = pandas.read_excel(path + "/power_data.xlsx")

    # Wavelength of spectrophotometer experiment
    wl = np.arange(400, 710, 10)

    # Finding centroid from green channel at 520 nm
    ind520 = 12

    # Reading image
    im_cen, met_cen = processing.readTIFF_xiMU(imlist[ind520])
    im_cen_dws = processing.dwnsampling(im_cen, "BGGR")
    im_cen_g = im_cen_dws[:, :, 1]

    # Region properties
    bina, regpro = processing.regionproperties(im_cen_g, 2E3)
    yc, xc = regpro[0].centroid

    yc = int(round(yc))
    xc = int(round(xc))
    print(yc, xc)

    square_px_number = 9

    interval = square_px_number//2

    DN_avg = np.empty((len(imlist), 3))
    DN_std = np.empty((len(imlist), 3))

    # Figure pre-allocation
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    fig2, ax2 = plt.subplots(1, 3, figsize=(12, 5))

    ax1.imshow(bina)
    ax1.plot(xc, yc, "r+", markersize=3)

    title = ["Red channel", "Green channel", "Blue channel"]

    #plt.ion()
    # Loop
    for n, impath in enumerate(imlist):
        print("Processing image number {0}".format(n))

        # Reading image
        im_op, met_op = processing.readTIFF_xiMU(impath)

        # Removing dark frame
        bl = int(met_op["black_level"])

        im_op -= bl

        # Downsampling
        im_dws = processing.dwnsampling(im_op, "BGGR")

        for i in range(im_dws.shape[2]):
            im = im_dws[:, :, i]

            DN_avg[n, i] = np.mean(im[yc-interval:yc+interval:1, xc-interval:xc+interval:1])
            DN_std[n, i] = np.std(im[yc-interval:yc+interval:1, xc-interval:xc+interval:1])

            # Rectangular patch
            rect = patches.Rectangle((xc-interval, yc-interval), square_px_number, square_px_number,
                                     linewidth=1, edgecolor='r', facecolor='none')
            # Figure
            ax2[i].imshow(im)
            ax2[i].set_title(title[i])

            # Axe in
            axins = zoomed_inset_axes(ax2[i], 3.5, loc=1)
            axins.imshow(im)
            axins.add_patch(rect)

            x1, x2, y1, y2 = xc-25, xc+25, yc-25, yc+25  # specify the limits
            axins.set_xlim(x1, x2)  # apply the x-limits
            axins.set_ylim(y1, y2)  # apply the y-limits

            mark_inset(ax2[i], axins, loc1=2, loc2=4, fc="none", ec="0", linestyle="--")

        #plt.pause(0.05)
    #plt.ioff()

    # Print results
    print(DN_avg)
    print(DN_std)

    # Relative standard deviation
    relative_std = (DN_std/DN_avg) * 100

    print(relative_std)

    # Relative spectral response
    pw = powerdata["puissance(nW)"]
    SP = DN_avg/pw[:, None]
    RSP = SP/np.max(SP, axis=0)

    # Interpolation at interval of 1 nm between 400 and 700 nm
    wl_interp = np.arange(400, 701, 1)

    RSP_interp = np.empty((len(wl_interp), 4), dtype=np.float32)
    RSP_interp[:, 0] = wl_interp
    RSP_interp[:, 1] = np.interp(wl_interp, wl, RSP[:, 0])
    RSP_interp[:, 2] = np.interp(wl_interp, wl, RSP[:, 1])
    RSP_interp[:, 3] = np.interp(wl_interp, wl, RSP[:, 2])

    # Finding peaks and FWHM of experimental results
    arg_peaks = np.argmax(RSP_interp[:, 1:4], axis=0)
    wl_peaks = RSP_interp[arg_peaks, 0]
    print(wl_peaks)

    results_half_r = np.array(peak_widths(RSP_interp[:, 1], (arg_peaks[0], ), rel_height=0.5))
    results_half_g = np.array(peak_widths(RSP_interp[:, 2], (arg_peaks[1], ), rel_height=0.5))
    results_half_b = np.array(peak_widths(RSP_interp[:, 3], (arg_peaks[2], ), rel_height=0.5))

    results_half_r[2:] = RSP_interp[np.around(results_half_r[2:]).astype(int), 0]
    results_half_g[2:] = RSP_interp[np.around(results_half_g[2:]).astype(int), 0]
    results_half_b[2:] = RSP_interp[np.around(results_half_b[2:]).astype(int), 0]

    results_half_r, results_half_g, results_half_b = tuple(results_half_r), tuple(results_half_g), tuple(results_half_b)

    FWHM = np.array([results_half_r[3] - results_half_r[2],
                    results_half_g[3] - results_half_g[2],
                    results_half_b[3] - results_half_b[2]])

    print(FWHM)

    # Finding peaks of manufacturer spectral response

    arg_peaks_manu = np.argmax(np.array(sensor_rsr_data[["R", "G", "B"]]), axis=0)
    wl_peaks_manu = np.array([sensor_rsr_data["RW"][arg_peaks_manu[0]],
                             sensor_rsr_data["GW"][arg_peaks_manu[1]],
                             sensor_rsr_data["BW"][arg_peaks_manu[2]]])

    print(wl_peaks_manu)

    # Figures
    # Beam power
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)

    ax3.plot(powerdata["wl "], powerdata["puissance(nW)"])

    ax3.set_xlabel("Wavelength [nm]")
    ax3.set_ylabel("Power [nW]")

    # Spectral response
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)

    ax4.plot(wl, RSP[:, 0], "r", linewidth=2, label="Experimental results")
    ax4.plot(wl, RSP[:, 1], "g", linewidth=2)
    ax4.plot(wl, RSP[:, 2], "b", linewidth=2)
    ax4.plot(wl_peaks_manu, np.array([1, 1, 1]), "x", color="grey")

    ax4.plot(sensor_rsr_data["RW"], sensor_rsr_data["R"], "k--", label="Manufacturer data")
    ax4.plot(sensor_rsr_data["GW"], sensor_rsr_data["G"], "k--")
    ax4.plot(sensor_rsr_data["BW"], sensor_rsr_data["B"], "k--")

    ax4.set_xlim((400, 700))
    ax4.set_ylim((0, 1.2))

    ax4.set_xlabel("Wavelength [nm]")
    ax4.set_ylabel("Relative spectral response")

    ax4.legend(loc="best")

    # Relative spectral response interpolated + maximum position + FWHM
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)

    ax5.plot(wl, RSP[:, 0], "r")
    ax5.plot(wl, RSP[:, 1], "g")
    ax5.plot(wl, RSP[:, 2], "b")

    ax5.plot(RSP_interp[:, 0], RSP_interp[:, 1], "ro")
    ax5.plot(RSP_interp[:, 0], RSP_interp[:, 2], "go")
    ax5.plot(RSP_interp[:, 0], RSP_interp[:, 3], "bo")

    ax5.plot(wl_peaks, np.array([1, 1, 1]), "x", color="grey")

    ax5.hlines(*results_half_r[1:], linestyles="-.", color="grey")
    ax5.hlines(*results_half_g[1:], linestyles="-.", color="grey")
    ax5.hlines(*results_half_b[1:], linestyles="-.", color="grey")

    ax5.set_xlabel("Wavelength [nm]")
    ax5.set_ylabel("Relative spectral response")

    # Saving results of calibration

    while True:
        inputsav = input("Do you want to save the calibration results? (y/n) : ")
        inputsav = inputsav.lower()
        if inputsav in ["y", "n"]:
            break

    if inputsav == "y":
        datetim = os.path.basename(imlist[0])
        name = "spectral_response_files/spectral_response_"
        np.savetxt(name + datetim[4:19] + ".csv",  np.c_[RSP_interp], header="wl,R,G,B", delimiter=',')

    plt.show()
