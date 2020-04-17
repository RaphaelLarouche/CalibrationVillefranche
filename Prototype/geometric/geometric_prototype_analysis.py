# -*- coding: utf-8 -*-
"""
Python file to perform geometric experiment analysis.
"""

# Importation of standard modules
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy import stats
import os
from scipy.optimize import minimize

# Importation of other modules
import cameracontrol.cameracontrol as cc


# Function
def residuals(theo_angles, fitted_angles, verbose=True):
    """
    Function that retrieves and print the residual of the fitted angles.
    :param theo_angles:
    :param fitted_angles:
    :return: linear regression parameters
    """

    shap = theo_angles.shape

    ta = abs(theo_angles.reshape(-1))
    fa = abs(fitted_angles.reshape(-1))

    linearfit = stats.linregress(ta[np.isfinite(ta)], fa[np.isfinite(fa)])

    ta = ta.reshape((-1, shap[1]))
    fa = fa.reshape((-1, shap[1]))

    residuals = abs(fa - (ta * linearfit[0] + linearfit[1]))

    if verbose:
        print("Residuals")
        print(residuals)
        print("Average residuals")
        print(np.nanmean(residuals, axis=0))
    return linearfit


if __name__ == "__main__":

    # *** Code beginning ***
    # Processing object
    processing = cc.ProcessImage()

    # impath = processing.folder_choice()

    # Creating a directory with all files needed
    # Separated files
    impath_00 = "/Volumes/KINGSTON/Villefranche/Prototype/Geometric/geometric_proto_20191115_2x2_00"
    impath_90 = "/Volumes/KINGSTON/Villefranche/Prototype/Geometric/geometric_proto_20191115_2x2_90"

    images_path_00 = glob.glob(impath_00 + "/IMG_*.tif")
    images_path_90 = glob.glob(impath_90 + "/IMG_*.tif")

    images_path_00.sort()
    images_path_90.sort()

    # All files
    impath = "/Volumes/KINGSTON/Villefranche/Prototype/Geometric/geometric_proto_20191115_4x4_[0-9][0-9]"
    images_path = glob.glob(impath + "/IMG_*.tif")
    images_path.sort()

    # Opening dark image
    image_dark_path_00 = glob.glob(impath_00 + "/DARK_*.tif")
    imdark_00, metdark_00 = processing.readTIFF_xiMU(image_dark_path_00[0])

    image_dark_path_90 = glob.glob(impath_90 + "/DARK_*.tif")
    imdark_90, metdark_90 = processing.readTIFF_xiMU(image_dark_path_90[0])

    imdark = {}
    imdark["0"] = imdark_00
    imdark["1"] = imdark_90

    # Opening geometric calibration results from chessboard experiment (Scaramuzza et al.)
    cwd = os.getcwd()
    #path_calib_cb_air = "/geometric_calibrationfiles_cb_air/geo_calibration_2x2_air_20191211_2152.npz"  # After moving CMOS !!!
    path_calib_cb_air = "/geometric_calibrationfiles_cb_air/geo_calibration_2x2_air_20191211_1714.npz"  # Before moving CMOS !!!
    geocalib = np.load(cwd + path_calib_cb_air, allow_pickle=True)
    print(geocalib.files)
    print("Center point chessboard calibration [x, y]")
    print(geocalib["centerpoint"])

    # Figure
    fig1 = plt.figure()  # Image addition + centroids
    ax1 = fig1.add_subplot(111)

    # Angles scanned
    angles = np.arange(-105, 110, 5)

    # Pre-allocation
    imtot = np.zeros((int(metdark_00["height"]), int(metdark_00["width"])))

    exposure = np.empty((len(images_path_00), 2)) * np.nan  # List of exposure time in second
    gain = np.empty((len(images_path_00), 2)) * np.nan
    good_angles = np.empty((len(images_path_00), 2)) * np.nan

    centro = np.empty((len(images_path_00), 2), dtype=[("y", "float32"), ("x", "float32")])
    centro.fill(np.nan)

    for n, path in enumerate(zip(images_path_00, images_path_90)):
        print("Processing image number {0}".format(n + 1))

        for j in range(len(path)):
            im, met = processing.readTIFF_xiMU(path[j])
            im -= imdark[str(j)]
            im = np.clip(im, 0, 2**12)

            # Addition of all the images
            imtot += im

            # Saving exposure time and gain
            exposure[n, j] = met["exposure_time_us"]
            gain[n, j] = met["gain_db"]

            # Region properties
            binary, regionprops = processing.regionproperties(im, 700)

            if regionprops:
                centro[n, j] = regionprops[0].centroid
                good_angles[n, j] = angles[n]
                ax1.plot(regionprops[0].centroid[1], regionprops[0].centroid[0], marker="+", color="r", markersize=4)

    # Clipping imtot
    imtot = np.clip(imtot, 0, 2**12)

    # plt.figure()
    # plt.imshow(processing.dwnsampling(imtot, "BGGR")[:, :, 0])
    # plt.show()

    # Printing results
    print(exposure)
    print(gain)
    print(good_angles)
    print(centro)

    xmean_tot = np.nanmean(centro["x"].reshape(-1))
    ymean_tot = np.nanmean(centro["y"].reshape(-1))

    # Optimized center --> 320, 282
    #xmean_tot, ymean_tot = int(320.41683008), int(282.75501102)

    # USING mean center
    dx_c, dy_c = centro["x"] - xmean_tot, centro["y"] - ymean_tot
    radial_c = np.sqrt(dx_c ** 2 + dy_c ** 2)

    # ___________________________________________________________________________
    # 2. Processing according to the data series average center (0 and 90 separated)
    xmean_00, xmean_90 = np.nanmean(centro["x"], axis=0)
    ymean_00, ymean_90 = np.nanmean(centro["y"], axis=0)

    # Azimuth 0
    dx_00, dy_00 = centro["x"][:, 0] - xmean_00, centro["y"][:, 0] - ymean_00
    radial_00 = np.sqrt(dx_00**2 + dy_00**2)

    # Azimuth 90
    dx_90, dy_90 = centro["x"][:, 1] - xmean_90, centro["y"][:, 1] - ymean_90
    radial_90 = np.sqrt(dx_90**2 + dy_90**2)
    # Fit of the projection function
    angles_tot_00 = good_angles[np.isfinite(good_angles[:, 0]), 0]
    angles_tot_90 = good_angles[np.isfinite(good_angles[:, 1]), 1]

    radial_tot_00 = radial_00[np.isfinite(radial_00)]
    radial_tot_90 = radial_90[np.isfinite(radial_90)]

    # Fit azimuth 0
    popt_00, pcov_00 = processing.geometric_curvefit(radial_tot_00, abs(angles_tot_00))
    radial_data_00 = np.linspace(0, max(radial_tot_00) + 10, 1000)
    # Fit azimuth 90
    popt_90, pcov_90 = processing.geometric_curvefit(radial_tot_90, abs(angles_tot_90))
    radial_data_90 = np.linspace(0, max(radial_tot_90) + 20, 1000)

    # Fit of all data (azimuth 0 and 90)
    #angle_tot = np.append(angles_tot_00[2:-2], angles_tot_90)   # from -90 to 90 for both
    #radial_tot = np.append(radial_tot_00[2:-2], radial_tot_90)  # from -90 to 90 for both
    angle_tot = np.append(angles_tot_00, angles_tot_90)
    radial_tot = np.append(radial_tot_00, radial_tot_90)

    popt_tot, pcov_tot = processing.geometric_curvefit(radial_tot, abs(angle_tot))
    radial_data_tot = np.linspace(0, max(radial_tot)+10, 1000)

    # ___________________________________________________________________________
    # Refitting angles from the centroid radial coordinate relative to the best mean center
    fitted_angles = processing.polynomial_fit(radial_c, *popt_tot)

    linfit = residuals(good_angles, fitted_angles)
    print(linfit)

    # Fitting best center
    center_init = np.array([xmean_tot, ymean_tot])
    res = minimize(processing.fit_imagecenter, center_init, method="Nelder-Mead", args=(centro, good_angles, popt_tot))
    print(res)

    xbest, ybest = res.x[0], res.x[1]
    radial_best = np.sqrt((centro["x"] - xbest)**2 + (centro["y"] - ybest)**2)
    radial_best = radial_best.reshape(-1)
    good_angles = good_angles.reshape(-1)

    radial_best = radial_best.reshape((-1, 2))
    good_angles = good_angles.reshape((-1, 2))
    radial_data_best = np.linspace(0, np.nanmax(radial_best)+10, 1000)

    fitted_angles_best = processing.polynomial_fit(radial_best, *popt_tot)

    linfit_best = residuals(good_angles, fitted_angles_best)
    print(linfit_best)

    # ___________________________________________________________________________
    # Computation of azimuth and zenith matrix

    imsize = imdark_00.shape
    print(imsize)

    zenith, azimuth = processing.angularcoordinates(imsize, center_init, popt_tot)

    cond = zenith > 110
    zenith[cond] = np.nan
    azimuth[cond] = np.nan

    # ___________________________________________________________________________
    # Figures
    # Figure 1 - image total
    ax1.imshow(imtot)
    ax1.axhline(ymean_00, 0, 647)
    ax1.axvline(xmean_90, 0, 500)
    ax1.plot(xmean_tot, ymean_tot, marker="o", markerfacecolor="None", markeredgecolor="k", markersize=4, label="Mean center position")
    ax1.plot(xbest, ybest, marker="s", markerfacecolor="None", markeredgecolor="y", markersize=4, label="Fitted center")
    ax1.plot(xmean_00, ymean_00, marker="o", markerfacecolor="None", markersize=4, label="Mean 0˚ azimuth")
    ax1.plot(xmean_90, ymean_90, marker="o", markerfacecolor="None", markersize=4, label="Mean 90˚ azimuth")

    ax1.legend(loc="best")

    # Figure 2 - radial distance vs. angle + fit
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    ax2.plot(radial_00, abs(good_angles[:, 0]), "d", markersize=3, markerfacecolor="None", markeredgecolor='red', markeredgewidth=1, label="0˚ azimuth")
    ax2.plot(radial_90, abs(good_angles[:, 1]), "s", markersize=3, markerfacecolor="None", markeredgecolor='blue', markeredgewidth=1, label="90˚ azimuth")

    #ax2.plot(radial_data_00, processing.polynomial_fit(radial_data_00, *popt_00), color="#1f77b4", label="Fit 0˚ azimuth")
    #ax2.plot(radial_data_90, processing.polynomial_fit(radial_data_90, *popt_90), color="#ff7f0e", label="Fit 90˚ azimuth")
    ax2.plot(radial_data_tot, processing.polynomial_fit(radial_data_tot, *popt_tot,), color="k", label="Polynomial fit all data")

    ax2.set_xlabel("Distance from image center [px]")
    ax2.set_ylabel("Plate angle [˚]")
    ax2.set_title("Relative to data series respective average positions")

    ax2.legend(loc="best")

    # Figure 3 - radial distance vs. angle for the mean center point
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)

    ax3.plot(radial_c[:, 0], abs(good_angles[:, 0]), "d", markersize=3, markerfacecolor="None", markeredgecolor='red', markeredgewidth=1, label="0˚ azimuth")
    ax3.plot(radial_c[:, 1], abs(good_angles[:, 1]), "s", markersize=3, markerfacecolor="None", markeredgecolor='blue', markeredgewidth=1, label="90˚ azimuth")

    ax3.set_xlabel("Distance from image center [px]")
    ax3.set_ylabel("Theoretical angle [˚]")
    ax3.set_title("Relative to overall average position")

    ax3.legend(loc="best")

    # Figure 4 - radial distance vs. angle for all data (0 and 90 azimuth)
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.plot(radial_best[:, 0], abs(good_angles[:, 0]), "d", markersize=3, markerfacecolor="None", markeredgecolor='red', markeredgewidth=1, label="0˚ azimuth")
    ax4.plot(radial_best[:, 1], abs(good_angles[:, 1]), "s", markersize=3, markerfacecolor="None", markeredgecolor='blue', markeredgewidth=1, label="90˚ azimuth")

    ax4.plot(radial_data_best, processing.polynomial_fit(radial_data_best, *popt_tot), color="k", label="Polynomial fit")
    ax4.plot(radial_data_best, processing.polynomial_fit_forcedzero(radial_data_best, *geocalib["fitparams"]), color="gray", linestyle="--", label="Chessboard calibration air")

    ax4.set_yticks(np.arange(-10, 120, 10))
    ax4.set_ylim([-5, 110])

    ax4.set_xlabel("Distance from image center [px]")
    ax4.set_ylabel("Theoretical angle [˚]")
    ax4.set_title("Relative to overall average position")

    ax4.legend(loc="best")

    # Figure 5 - theoric vs refit angle
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)

    legend_regression = "slope: {0:.3f}\nintercept: {1:.3f}\nr-squared: {2:.3f}".format(linfit_best[0], linfit_best[1], linfit_best[2]**2)

    ax5.plot(abs(good_angles[:, 0]), abs(fitted_angles_best[:, 0]), "d", markersize=3, markerfacecolor="None", markeredgecolor='red', markeredgewidth=1, label="0˚ azimuth")
    ax5.plot(abs(good_angles[:, 1]), abs(fitted_angles_best[:, 1]), "s", markersize=3, markerfacecolor="None", markeredgecolor='blue', markeredgewidth=1, label="90˚ azimuth")
    ax5.plot(np.linspace(0, 110, 1000), linfit_best[0]*np.linspace(0, 110, 1000)+linfit_best[1], 'k', label="Linear regression")
    ax5.text(0, 50, legend_regression)

    ax5.set_ylabel("Fit angle [˚]")
    ax5.set_xlabel("Theoretical angle [˚]")

    ax5.legend(loc="best")

    # Figure 6
    fig6 = plt.figure()
    ax6_1, ax6_2 = fig6.add_subplot(121), fig6.add_subplot(122)

    ax6_1.imshow(azimuth)
    ax6_2.imshow(zenith)

    # Saving figures
    fig1.savefig("/Users/raphaellarouche/Desktop/CalibVillefranche/fig1.eps")
    fig2.savefig("/Users/raphaellarouche/Desktop/CalibVillefranche/fig2.eps")
    fig3.savefig("/Users/raphaellarouche/Desktop/CalibVillefranche/fig3.eps")
    fig5.savefig("/Users/raphaellarouche/Desktop/CalibVillefranche/fig5.eps")

    # Saving results of calibration
    while True:
        inputsav = input("Do you want to save the calibration results? (y/n) : ")
        inputsav = inputsav.lower()
        if inputsav in ["y", "n"]:
            break

    if inputsav == "y":
        if "4x4" in impath_00 and impath_90:
            name = "geo_calibration_results_4x4_"
        elif "2x2" in impath_00 and impath_90:
            name = "geo_calibration_results_2x2_"

        savename = "geometric_calibrationfiles_air/" + name + impath_00[-15:-7] + ".npz"
        np.savez(savename, imagesize=imsize, centerpoint=center_init, fitparams=popt_tot)

    plt.show()
