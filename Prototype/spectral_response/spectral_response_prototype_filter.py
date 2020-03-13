# -*- coding: utf-8 -*-
"""
Python script to process data taken with spectrophotometer Perkin Elmer Lambda 850 and retrieve relative spectral
response of the prototype. 2 nm beam bandwidth and short step (5 nm or 1 nm).
"""

if __name__ == "__main__":

    # Importation of standard modules
    import numpy as np
    import matplotlib.pyplot as plt
    import glob
    import pandas
    import os
    import time

    # Importation of other modules
    import cameracontrol.cameracontrol as cameracontrol

    # ___________________________________________________________________________
    # *** Code beginning ***

    processing = cameracontrol.ProcessImage()

    # Opening simulation curves (measured spectral response without filter + manufacturer data from their site
    data_spectral_fitlers = pandas.read_csv("~/PycharmProjects/simulation_spectral_error/RSR_shifted_interpolated/5066-asci.csv", sep=",")
    print(data_spectral_fitlers.keys())

    # Image path
    path = "/Volumes/KINGSTON/Quebec/Prototype/Spectral_response/spectral_response_2x2_20200304"
    imagelist = glob.glob(path + "/IMG_*tif")
    imagelist.sort()

    # Dark
    imagelistdark = glob.glob(path + "/DARK*.tif")

    # Wavelength range for this experiment (to be changed)
    wl = np.arange(400, 705, 5)
    print(wl)

    # Interpolation of the beam power data
    powerdata = pandas.read_excel(path + "/power_data.xlsx")
    creation_time = time.ctime(os.stat(path + "/power_data.xlsx").st_ctime)
    powerdata_interpo = processing.interpolation(wl, (powerdata["wl "], powerdata["puissance(nW)"]))

    # Find centroid of the beam
    nm = 470
    ind = int(np.where(wl == nm)[0])
    im_cen, met_cen = processing.readTIFF_xiMU(imagelist[ind])
    im_cen_b = processing.dwnsampling(im_cen, "BGGR")[:, :, 2]

    bina, regpro = processing.regionproperties(im_cen_b, 1E3)
    yc, xc = regpro[0].centroid
    yc = int(round(yc))
    xc = int(round(xc))
    print("centroid x: {0:d}, centroid y: {1:d}".format(xc, yc))

    # LOOP
    # Pre-allocation
    square_pix_avg = 9
    interval = square_pix_avg//2

    DN_avg = np.empty((len(imagelist), 3))
    DN_std = np.empty((len(imagelist), 3))

    # Dark
    im_dk, met_dk = processing.readTIFF_xiMU(imagelistdark[0])

    for n, impath in enumerate(imagelist):
        print("Processing image number {0}".format(n))

        # Reading image number
        im_op, met_op = processing.readTIFF_xiMU(impath)

        # Removing dark
        im_op -= im_dk

        # Downsampling
        im_dws = processing.dwnsampling(im_op, "BGGR")

        for i in range(im_dws.shape[2]):
            im = im_dws[:, :, i]

            DN_avg[n, i] = np.mean(im[yc-interval:yc+interval:1, xc-interval:xc+interval:1])
            DN_std[n, i] = np.std(im[yc-interval:yc+interval:1, xc-interval:xc+interval:1])

    # Printing results
    print(DN_avg)

    relative_std = (DN_std / DN_avg) * 100
    print(relative_std)

    # Relative spectral response
    SP = DN_avg / powerdata_interpo[:, None]
    RSP = SP / np.max(SP, axis=0)

    # ___________________________________________________________________________
    # Figures
    # Figure 1
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    ax1.imshow(im_cen_b)
    ax1.imshow(bina, alpha=0.2)
    ax1.plot(xc, yc, "r+")

    # Figure 2
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    ax2.plot(powerdata["wl "], powerdata["puissance(nW)"], label="Experimental data: {0}".format(creation_time))
    ax2.plot(wl, powerdata_interpo, 'ro', label="Interpolation each 5 nm")

    ax2.set_xlabel("Wavelength [nm]")
    ax2.set_ylabel("Power [nW]")
    ax2.legend(loc="best")

    # Figure 3
    labe = ["Red band", "Green band", "Blue band"]
    fig3, ax3 = plt.subplots(3, 1, sharex=True, figsize=(8, 7))

    for n, a in enumerate(ax3):
        a.plot(wl, RSP[:, n], label=labe[n])
        a.plot(data_spectral_fitlers[data_spectral_fitlers.keys()[0]],
               data_spectral_fitlers[data_spectral_fitlers.keys()[n+1]], "k--", label="Manufacturer data")
        a.legend(loc="best")
        a.set_ylabel("RSR")

        if n == 2:
            a.set_xlabel("Wavelength [nm]")

    plt.show()
