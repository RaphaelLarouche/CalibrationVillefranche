# -*- coding: utf-8 -*-
"""
Python file for roll-off data analysis for the prototype with 2x2 binning (integrating sphere).
"""

if __name__ == "__main__":

    # Importation of standard modules
    import numpy as np
    import glob
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    # Importation of other modules
    import cameracontrol.cameracontrol as cc

    # Function

    def roff_fitcurve(x, a0, a2, a4, a6, a8):
        """

        :param x:
        :param a0:
        :param a2:
        :param a4:
        :param a6:
        :param a8:
        :return:
        """
        return a0 + a2*x**2 + a4*x**4 + a6*x**6 + a8*x**8

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


    # *** Code beginning ***
    processing = cc.ProcessImage()

    # Image directory creation
    #path_00 = "/Volumes/KINGSTON/Quebec/Prototype/Rolloff/rolloff_proto_air/rolloff_proto_20191213_2x2"
    path_00 = "/Volumes/KINGSTON/Quebec/Prototype/Rolloff/rolloff_proto_air/rolloff_proto_20200117_2x2_02"
    images_path_00 = glob.glob(path_00 + "/IMG_*.tif")
    images_path_00.sort()

    # Opening dark image
    image_path_dark = glob.glob(path_00 + "/DARK*.tif")
    image_path_dark.sort()
    imdark_00, metdark_00 = processing.readTIFF_xiMU(image_path_dark[0])

    # ___________________________________________________________________________
    # Loop to get roll=off curves

    # Pre-allocation
    # Figure roll-off
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)

    imtotal_00 = np.zeros((imdark_00.shape[0] // 2, imdark_00.shape[1] // 2, 3))

    centroids_00 = np.empty((len(images_path_00), 3), dtype=[("y", "float32"), ("x", "float32")])
    centroids_00.fill(np.nan)

    rolloff_00 = np.empty((len(images_path_00), 3))
    rolloff_00.fill(np.nan)

    std_rolloff_00 = np.empty((len(images_path_00), 3))
    std_rolloff_00.fill(np.nan)

    #angles = np.arange(-100, 105, 5)
    angles = np.arange(-105, 110, 5)

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

                ROI = im_dws[yc-1:yc+2:1, xc-1:xc+2:1, i]

                rolloff_00[n, i] = np.mean(ROI)
                std_rolloff_00[n, i] = np.std(ROI)

                print(ROI.shape)

    # Printing results
    print(rolloff_00)
    print(std_rolloff_00)

    # Relative standard error
    RSE = std_rolloff_00 / rolloff_00[None, :]

    RSE = np.squeeze(RSE)

    # Normalization of roll-off
    rolloff_00 = rolloff_00 / np.nanmax(rolloff_00, axis=0)
    print(rolloff_00)

    # Display RSE
    print(RSE)

    # Sorting roll-off
    ind_sroff = np.tile(np.argsort(abs(angles)), (3, 1)).T

    print(ind_sroff[:, 0])

    sangle = np.sort(abs(angles))
    sroff = np.take_along_axis(rolloff_00, ind_sroff, axis=0)
    sRSE = np.take_along_axis(RSE,  ind_sroff, axis=0)

    # Fit
    xdata = np.linspace(0, 105, 1000)
    col = ["r", "g", "b"]
    bandnames = {"r": "red channel", "g": "green channel", "b" : "blue channel"}
    for n in range(rolloff_00.shape[1]):
        val = rolloff_00[:, n]
        ang = abs(angles)

        mask = ~np.isnan(val)

        val = val[mask]
        ang = ang[mask]
        popt, pcov = curve_fit(roff_fitcurve, ang, val)

        # Standard deviation of estimated parameters
        perr = np.sqrt(np.diag(pcov))

        # Display polynomial coefficients, their std and the determination coefficient
        print(bandnames[col[n]])
        disp_roll_fitcurve(popt, perr)

        residuals = val - roff_fitcurve(ang, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((val - np.mean(val))**2)

        rsquared = 1 - (ss_res / ss_tot)
        print(rsquared)

        ax3.plot(xdata, roff_fitcurve(xdata, *popt), color=col[n], alpha=0.7, label="Polynomial fit {0} ($R^2$={1:.3f})".format(bandnames[col[n]], rsquared))

    # Figure configuration
    # Figure 1 - Image of output port of integrating sphere
    imtotal_00 = np.clip(imtotal_00, 0, 2**12)

    for j in range(imtotal_00.shape[2]):
        ax1[j].imshow(imtotal_00[:, :, j])

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

    plt.show()
