# -*- coding: utf-8 -*-
"""
Python file to performed geometric experiment analysis for Insta360 ONE camera.
"""

if __name__ == "__main__":

    # Importation of standard modules
    import matplotlib.pyplot as plt
    import numpy as np
    import glob
    from scipy import stats
    import pandas
    from scipy.optimize import minimize

    # Importation of other modules
    import cameracontrol.cameracontrol as cc

    # *** Code beginning ***
    # Processing object
    processing = cc.ProcessImage()

    # Reading results for calibration done with chessboard target
    cboardcalib = pandas.read_csv("/Users/raphaellarouche/Desktop/CalibVillefranche/chessboardfitcmp.csv", delimiter=",")
    print(cboardcalib)

    # Creating directories for files
    # LensClose
    #path_00deg = "/Volumes/KINGSTON/Villefranche/Insta360/Geometric/LensClose/20191112_geometric/20191112_geometric_00_05"
    #path_90deg = "/Volumes/KINGSTON/Villefranche/Insta360/Geometric/LensClose/20191121_geometric/20191121_geometric_90_01"

    # LensFar
    path_00deg = "/Volumes/KINGSTON/Villefranche/Insta360/Geometric/LensFar/20191121_geometric/20191121_geometric_00_01"
    path_90deg = "/Volumes/KINGSTON/Villefranche/Insta360/Geometric/LensFar/20191121_geometric/20191121_geometric_90_01"

    # 1. Data
    images_path_00deg = glob.glob(path_00deg + "/IMG_*.dng")
    images_path_90deg = glob.glob(path_90deg + "/IMG_*.dng")

    images_path_00deg.sort()
    images_path_90deg.sort()

    # 2. Darkframes
    image_path_dark00deg = glob.glob(path_00deg + "/DARK_*.dng")
    image_path_dark90deg = glob.glob(path_90deg + "/DARK_*.dng")

    image_path_dark00deg.sort()
    image_path_dark90deg.sort()

    # Automatic detection of which image
    wlens = "far"
    if "LensClose" in images_path_00deg[0] and images_path_90deg[0]:
        wlens = "close"

    # Opening darkframe
    imdark_00deg, metdark_00deg = processing.readDNG_insta360(image_path_dark00deg[0], which_image=wlens)
    imdark_90deg, _ = processing.readDNG_insta360(image_path_dark90deg[0], which_image=wlens)

    imdark = np.empty((int(metdark_00deg["Image ImageLength"].values[0]/2),
                       int(metdark_00deg["Image ImageWidth"].values[0]), 2), dtype="int64")

    imdark[:, :, 0] = imdark_00deg
    imdark[:, :, 1] = imdark_90deg

    # Angles scanned
    angles = np.arange(-90, 95, 5)

    # Pre-allocation
    imtot = np.zeros((int(metdark_00deg["Image ImageLength"].values[0]/2), metdark_00deg["Image ImageWidth"].values[0]))
    imtot_binary = np.zeros((int(metdark_00deg["Image ImageLength"].values[0]/2), metdark_00deg["Image ImageWidth"].values[0]))

    exposure = np.empty((len(images_path_00deg), 2)) * np.nan  # List of exposure time in second
    gain = np.empty((len(images_path_00deg), 2)) * np.nan    # List of gain in ISO
    good_angles = np.empty((len(images_path_00deg), 2)) * np.nan

    centro = np.empty((len(images_path_00deg), 2), dtype=[("y", "float32"), ("x", "float32")])  # List of centroids
    centro.fill(np.nan)

    # Figure
    fig1 = plt.figure()  # Image addition + centroids
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure()  # Binary image addition + centroids
    ax2 = fig2.add_subplot(111)

    # Looping over images
    for n, path in enumerate(zip(images_path_00deg, images_path_90deg)):
        print("Processing image number {0}".format(n+1))

        for j in range(len(path)):
            # Reading data
            im, met = processing.readDNG_insta360(path[j], wlens)
            im -= imdark[:, :, j]  # Dark frame substraction
            im = np.clip(im, 0, 2**14)  # Clipping

            # Addition of all the images
            imtot += im

            # Region properties
            if wlens == "far":
                binary, regionprops = processing.regionproperties(im, 0.1E4) #LensFar
            elif wlens == "close":
                binary, regionprops = processing.regionproperties(im, 0.9E4)  #LenClose

            if regionprops:
                # Storing exposure time and gain
                exposure[n, j] = processing.ratio2float([met["Image ExposureTime"].values[0]])[0]
                gain[n, j] = met["Image ISOSpeedRatings"].values[0]
                good_angles[n, j] = angles[n]

                # Saving centroids
                centro[n, j] = regionprops[0].centroid

                # Plot of centroid
                ax1.plot(regionprops[0].centroid[1], regionprops[0].centroid[0], marker="+", color="r", markersize=4)
                ax2.plot(regionprops[0].centroid[1], regionprops[0].centroid[0], marker="+", color="r", markersize=4)
                imtot_binary += binary

    # Clipping imtot
    imtot = np.clip(imtot, 0, 2 ** 14)

    # Printing results
    print(exposure)
    print(gain)
    print(good_angles)
    print(centro)

    # ___________________________________________________________________________
    # 1. Processing according to the overall average image center (0 and 90 together)
    # Finding image center using the average of all x, y position
    #xmean, ymean = np.nanmean(centro["x"].reshape(-1)), np.nanmean(centro["y"].reshape(-1))
    #print(xmean, ymean)

    # New best mean
    if wlens == "far":
        xmean, ymean = int(1756.70464206), int(1693.42991563)  # Lens Far
    elif wlens == "close":
        xmean, ymean = int(1698.7447675), int(1755.53913642)  # Lens Close

    # Radial distance for all the centroids according to the image center
    dx_imc, dy_imc = centro["x"] - xmean, centro["y"] - ymean
    radial_imc = np.sqrt(dx_imc**2 + dy_imc**2)

    # Fit for all data (0 an 90 degrees azimuth) using the average found image center
    radial_imc_tot = radial_imc.reshape(-1)
    good_angles_tot = good_angles.reshape(-1)

    pop_tot, pcov_tot = processing.geometric_curvefit(radial_imc_tot[np.isfinite(radial_imc_tot)], abs(good_angles_tot[np.isfinite(good_angles_tot)]))
    radial_data_tot = np.linspace(0, max(radial_imc_tot)+10, 1000)

    # ___________________________________________________________________________
    # 2. Processing according to the data series average center (0 and 90 separated)
    xmean_00, xmean_90 = np.nanmean(centro["x"], axis=0)
    ymean_00, ymean_90 = np.nanmean(centro["y"], axis=0)

    # Azimuth 0
    dx_00, dy_00 = centro["x"][:, 0] - xmean_00, centro["y"][:, 0] - ymean_00
    angles_tot_00 = good_angles[np.isfinite(good_angles[:, 0]), 0]
    radial_00 = np.sqrt(dx_00 ** 2 + dy_00 ** 2)
    radial_00_tot = radial_00[np.isfinite(radial_00)]

    # Azimuth 90
    dx_90, dy_90 = centro["x"][:, 1] - xmean_90, centro["y"][:, 1] - ymean_90
    angles_tot_90 = good_angles[np.isfinite(good_angles[:, 1]), 1]
    radial_90 = np.sqrt(dx_90**2 + dy_90**2)
    radial_90_tot = radial_90[np.isfinite(radial_90)]

    angle_tot_best = np.append(angles_tot_00, angles_tot_90)
    radial_tot_best = np.append(radial_00_tot, radial_90_tot)

    popt_best, pcov_best = processing.geometric_curvefit(radial_tot_best, abs(angle_tot_best))
    radial_data_best = np.linspace(0, max(radial_tot_best)+10, 1000)

    # ___________________________________________________________________________
    # Refitting the coordinates of the points
    refitted_angles = processing.polynomial_fit(radial_imc, *popt_best)

    A1 = abs(good_angles_tot)
    A2 = abs(refitted_angles.reshape(-1))

    linear_regress = stats.linregress(abs(A1[np.isfinite(A1)]), A2[np.isfinite(A2)])
    print(linear_regress)

    A1 = A1.reshape((len(images_path_00deg), 2))
    A2 = A2.reshape((len(images_path_00deg), 2))

    residuals = abs(A2 - (A1 * linear_regress[0] + linear_regress[1]))
    print("Residuals")
    print(residuals)
    print("Average residuals")
    print(np.nanmean(residuals, axis=0))

    # Refinement of the image center
    center_init = np.array([xmean, ymean])
    refinement_imcenter = minimize(processing.fit_imagecenter, center_init, method="Nelder-Mead",
                                   args=(centro, good_angles, popt_best))
    print(refinement_imcenter)

    # ___________________________________________________________________________
    # Computation of azimuth and zenith matrix

    imsize = imdark_00deg.shape

    zenith, azimuth = processing.angularcoordinates(imsize, center_init, popt_best)

    cond = zenith > 90
    zenith[cond] = np.nan
    azimuth[cond] = np.nan

    #___________________________________________________________________________
    # Figures
    # Figure 1 - image tot
    ax1.imshow(imtot)
    ax1.axhline(ymean, 0, 3456)
    ax1.axvline(xmean, 0, 3456)
    ax1.plot(xmean, ymean, marker="o", markerfacecolor="None", markeredgecolor="k", markersize=4,
             label="Image center position")
    ax1.plot(xmean_00, ymean_00, marker="o", markerfacecolor="None", markersize=4, label="Average position 0˚ azimuth")
    ax1.plot(xmean_90, ymean_90, marker="o", markerfacecolor="None", markersize=4, label="Average position 90˚ azimuth")

    ax1.legend(loc="best")

    # Figure 2 - image tot binary
    ax2.imshow(imtot_binary)
    ax2.axhline(ymean, 0, 3456)
    ax2.axvline(xmean, 0, 3456)
    ax2.plot(xmean, ymean, marker="o", markerfacecolor="None", markeredgecolor="k", markersize=4,
             label="Image center position")
    ax2.plot(xmean_00, ymean_00, marker="o", markerfacecolor="None", markersize=4, label="Average position 0˚ azimuth")
    ax2.plot(xmean_90, ymean_90, marker="o", markerfacecolor="None", markersize=4, label="Average position 90˚ azimuth")

    ax2.legend(loc="best")

    # Figure 3 - radial distance vs. angle for the average center point
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)

    ax3.plot(radial_imc[:, 0], abs(good_angles[:, 0]), "d", markersize=3, markerfacecolor="None", markeredgecolor='red', markeredgewidth=1, label="0˚ azimuth")
    ax3.plot(radial_imc[:, 1], abs(good_angles[:, 1]), "s", markersize=3, markerfacecolor="None", markeredgecolor='blue', markeredgewidth=1, label="90˚ azimuth")
    ax3.plot(radial_data_tot, processing.polynomial_fit(radial_data_tot, *pop_tot), "k", label="Polynomial fit")
    if wlens == "far":
        ax3.plot(cboardcalib["R"], cboardcalib["theta_LF"], "-.", label="Calibration with chessboard")
    elif wlens == "close":
        ax3.plot(cboardcalib["R"], cboardcalib["theta_LC"], "-.", label="Calibration with chessboard")

    ax3.set_xlabel("Distance from image center [px]")
    ax3.set_ylabel("Theoretical angle [˚]")
    ax3.set_title("Relative to overall average position")

    ax3.legend(loc="best")

    # Figure 4 - radial distance vs. angle  for the average center point of data series respectively
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)

    ax4.plot(radial_00, abs(good_angles[:, 0]), "d", markersize=3, markerfacecolor="None", markeredgecolor='red', markeredgewidth=1, label="0˚ azimuth")
    ax4.plot(radial_90, abs(good_angles[:, 1]), "s", markersize=3, markerfacecolor="None", markeredgecolor='blue', markeredgewidth=1, label="90˚ azimuth")
    ax4.plot(radial_tot_best, processing.polynomial_fit(radial_tot_best, *popt_best), "k", label="Polynomial fit all data")
    if wlens == "far":
        ax4.plot(cboardcalib["R"], cboardcalib["theta_LF"], "-.", label="Chessboard calibration air")
    elif wlens == "close":
        ax4.plot(cboardcalib["R"], cboardcalib["theta_LC"], "-.", label="Chessboard calibration air")

    ax4.set_xlabel("Distance from image center [px]")
    ax4.set_ylabel("Theoretical angle [˚]")
    ax4.set_title("Relative to data series respective average positions")

    ax4.legend(loc="best")

    # Figure 5 - refit of angles and comparisons
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)

    legend_regression = "slope: {0:.3f}\nintercept: {1:.3f}\nr-squared: {2:.3f}".format(linear_regress[0], linear_regress[1], linear_regress[2] ** 2)
    datax_linreg = np.linspace(0, 95, 1000)

    ax5.plot(abs(good_angles[:, 0]), refitted_angles[:, 0], "d", markersize=3, markerfacecolor="None", markeredgecolor='red', markeredgewidth=1, label="0˚ azimuth")
    ax5.plot(abs(good_angles[:, 1]), refitted_angles[:, 1], "s", markersize=3, markerfacecolor="None", markeredgecolor='blue', markeredgewidth=1, label="90˚ azimuth")
    ax5.plot(datax_linreg, linear_regress[0] * datax_linreg + linear_regress[0], "k", label="Linear regression")
    ax5.text(0, 40, legend_regression)

    ax5.set_xlabel("Theoretical angle [˚]")
    ax5.set_ylabel("Fit angle [˚]")

    ax5.legend(loc="best")

    # Figure - zenith and azimuth
    fig6 = plt.figure()
    ax6_1, ax6_2 = fig6.add_subplot(121), fig6.add_subplot(122)

    ax6_1.imshow(azimuth)
    ax6_2.imshow(zenith)

    # Saving
    # Saving figure
    fig1.savefig("/Users/raphaellarouche/Desktop/CalibVillefranche/ComLenClosefig1.eps")
    fig3.savefig("/Users/raphaellarouche/Desktop/CalibVillefranche/ComLenClosefig3.eps")
    fig4.savefig("/Users/raphaellarouche/Desktop/CalibVillefranche/ComLenClosefig4.eps")
    fig5.savefig("/Users/raphaellarouche/Desktop/CalibVillefranche/ComLenClosefig5.eps")

    # Saving results of calibration
    while True:
        inputsav = input("Do you want to save the calibration results? (y/n) : ")
        inputsav = inputsav.lower()
        if inputsav in ["y", "n"]:
            break

    if inputsav == "y":
        if wlens == "far":
            savename = "geometric_calibrationfiles_air/geo_LensFar_calibration_results_" + path_00deg[-24:-16] + ".npz"
        elif wlens == "close":
            savename = "geometric_calibrationfiles_air/geo_LensClose_calibration_results_" + path_00deg[-24:-16] + ".npz"
        np.savez(savename, imagesize=imsize, centerpoint=center_init, fitparams=popt_best)

    plt.show()
