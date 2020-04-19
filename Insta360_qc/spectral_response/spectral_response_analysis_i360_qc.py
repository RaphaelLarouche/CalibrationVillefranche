# -*- coding: utf-8 -*-
"""
Python script to plot data of Insta360 ONE spectral response
"""

# Importation of standard modules
import scipy.io
import pandas
import numpy as np
import matplotlib.pyplot as plt



# Importation of other modules

if __name__ == "__main__":

    # Opening data
    pathname_spectral_response = "/Users/raphaellarouche/Documents/MATLAB/radiance_cam_insta360/" \
                                 "characterization_spectral_response/characterization_spectral_response_files/"

    spectral_response = scipy.io.loadmat(pathname_spectral_response + "spectral_response_05_17_2019.mat")
    spectral_response = spectral_response["cmos_sensor_data"]

    # Opening reference data
    pathIMX377 = "/Users/raphaellarouche/Desktop/Sony_CMOS_Spectral_Response/IMX377.csv"
    refIMX377 = pandas.read_csv(pathIMX377, header=None)

    # Figures
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    wl = spectral_response["wavelength"][0][0]
    QE = spectral_response["QE"][0][0]

    color = iter(['#d62728', '#2ca02c', '#1f77b4'])
    lab = ["red pixels", "green pixels", "blue pixels"]

    indx = np.where(np.diff(refIMX377.loc[:, 0]) < 0)
    indx2 = np.zeros(4)
    indx2[1:3] = indx[0]
    indx2[3] = refIMX377.loc[:, 0].shape[0]

    for band in range(QE.shape[1]):

        cl = next(color)
        ax1.plot(wl, QE[:, band]/100, color=cl, linewidth=1.6, label=lab[band])

        effective_bw = np.trapz(QE[:, band]/100, x=wl[:, 0])
        effective_wl = np.trapz((QE[:, band]/100) * wl[:, 0], x=wl[:, 0])/effective_bw

        print("Effective bw {0}: {1:.4f}".format(lab[band], effective_bw))
        print("Effective wl {0}: {1:.4f}".format(lab[band], effective_wl))

        ax1.plot(refIMX377.loc[indx2[band]+1:indx2[band+1], 0], refIMX377.loc[indx2[band]+1:indx2[band+1], 1], "k-.")

    ax1.plot(np.nan, np.nan, "k-.", label="Sony IMX 377")

    ax1.set_xlabel("Wavelength [nm]", fontsize=13)
    ax1.set_ylabel("RSR", fontsize=13)

    ax1.legend(loc="best", fontsize=9)

    fig1.savefig("/Users/raphaellarouche/Desktop/Figures_pres20avril/rsr_com.eps", dpi=600)

    plt.show()
