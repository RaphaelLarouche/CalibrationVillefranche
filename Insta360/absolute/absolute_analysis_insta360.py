# -*- coding: utf-8 -*-
"""
Python file to process absolute radiance calibration data for the prototype.
"""

if __name__ == "__main__":

    # Importation of standard modules
    import numpy as np
    import matplotlib.pyplot as plt
    import glob
    import os
    import pandas
    from scipy.io import loadmat

    # Importation of other modules
    import cameracontrol.cameracontrol as cc

    # *** Code beginning ***
    # Creating object from class ProcessImage
    processing = cc.ProcessImage()

    # Input lens to be analyzed
    while True:
        answer = input("Which lens do you want to analyze? (c/f): ")
        if answer.lower() in ["c", "f"]:
            break
    # Different usable path
    di = os.path.dirname(os.getcwd())
    pathname_matlab = "/Users/raphaellarouche/Documents/MATLAB/radiance_cam_insta360/"
    generalpath = "~/PycharmProjects/CalibrationVillefranche/"
    if answer.lower() == "c":
        # Parameter to open the images
        wlens = "close"

        # Path lens close
        impath = "/Volumes/KINGSTON/Villefranche/Insta360/Absolute/LensClose/absolute_20191120/absolute_20191120_1ov60"

        # Geometric calibration
        geocal = np.load(di + "/geometric/geometric_calibrationfiles_air/geo_LensClose_calibration_results_20191112.npz")

        # Central point according to chessboard target calibration
        cpoint = np.array([1731 - 1, 1711 - 1])

        # Calibration coefficient with integrating sphere
        calib_coeff_sphere = loadmat(pathname_matlab + "calibration_absolute_radiance/calibration_absolute_radiance_files/C_Lcoeff_close_04_24_2019.mat")
        calib_coeff_sphere = calib_coeff_sphere['L_coeff_close'].reshape(-1)

    elif answer.lower() == "f":
        # Parameter to open the images
        wlens = "far"

        # Path lens far
        impath = "/Volumes/KINGSTON/Villefranche/Insta360/Absolute/LensFar/absolute_20191120/absolute_20191120_1ov60"

        # Geometric calibration
        geocal = np.load(di + "/geometric/geometric_calibrationfiles_air/geo_LensFar_calibration_results_20191121.npz")

        # Central point according to chessboard target calibration
        cpoint = np.array([1739 - 1, 1716 - 1])

        # Calibration coefficient with integrating sphere
        calib_coeff_sphere = loadmat(pathname_matlab + "calibration_absolute_radiance/calibration_absolute_radiance_files/C_Lcoeff_far_04_24_2019.mat")
        calib_coeff_sphere = calib_coeff_sphere["L_coeff_far"].reshape(-1)

    # List of image path
    image_path = glob.glob(impath + "/IMG_*.dng")
    ambiant_path = glob.glob(impath + "/DARK_*.dng")

    image_path.sort()
    ambiant_path.sort()

    image_path = image_path

    # Opening dark image
    ambiant, metambiant = processing.readDNG_insta360(ambiant_path[0], which_image=wlens)

    # ___________________________________________________________________________
    # Calculation of average spectral radiance in each channel

    # Opening spectral radiance of the calibration source
    sphere_source = loadmat("/Users/raphaellarouche/Documents/MATLAB/LuminanceAbsolueTests/TL_source_rad_04242019.mat")
    sphere_source = sphere_source["TL_source_rad"][0][0]
    print(sphere_source["lum_mean"])

    # Spectral response for both sensor of Insta360
    data_spectral = loadmat(pathname_matlab + "characterization_spectral_response/characterization_spectral_response_files/spectral_response_05_17_2019.mat")
    wl_sensor = data_spectral["cmos_sensor_data"]["wavelength"]
    qe_sensor = data_spectral["cmos_sensor_data"]["QE"]

    wl_sensor = np.array(wl_sensor[0][0][::10])
    qe_sensor = np.array(qe_sensor[0][0][::10])

    qe_norm = qe_sensor / np.max(qe_sensor, axis=0)

    # Bandwidth
    bwr = np.trapz(qe_norm[:, 0], x=wl_sensor[:, 0])
    bwg = np.trapz(qe_norm[:, 1], x=wl_sensor[:, 0])
    bwb = np.trapz(qe_norm[:, 2], x=wl_sensor[:, 0])

    # Opening lamp irradiance data

    irrfile = pandas.read_excel(generalpath + "files/FEL_GS_1015_opt.xlsx")
    irrfile = irrfile.where((400 <= irrfile["wavelength [nm]"]) & (irrfile["wavelength [nm]"] <= 700))
    irrfile = irrfile.dropna()

    wl_irr = irrfile["wavelength [nm]"]
    irr_50 = irrfile["absolute irradiance [mW m-2 nm-1]"]
    distance_lamp = 179.9  # cm

    reflectancefile = pandas.read_csv(generalpath + "files/spectralon.txt", delimiter="\t")
    reflectancefile = reflectancefile[::10]

    reflectancefile = reflectancefile.where((400 <= reflectancefile["Wl"]) & (reflectancefile["Wl"] <= 700))
    reflectancefile = reflectancefile.dropna()
    reflectance = reflectancefile["R"]
    wl_refl = reflectancefile["Wl"]

    irr = irr_50 * (50/distance_lamp)**2

    # Computation of radiance
    radiance = (irr * np.array(reflectance)*0.001)/np.pi

    # Computation of radiance convoluted with normalized spectral response
    radiance_conv = qe_norm * np.tile(np.array(radiance), (3, 1)).T

    # Integration for average radiance
    intr = np.trapz(radiance_conv[:, 0], x=wl_irr) / bwr
    intg = np.trapz(radiance_conv[:, 1], x=wl_irr) / bwg
    intb = np.trapz(radiance_conv[:, 2], x=wl_irr) / bwb

    ave_radiance = np.array([intr, intg, intb])
    print(ave_radiance)

    # ___________________________________________________________________________
    # Angular coordinates
    zenith, azimuth = processing.angularcoordinates(geocal["imagesize"], cpoint, geocal["fitparams"])
    cond = zenith > 90
    zenith[cond] = np.nan
    azimuth[cond] = np.nan

    zenith, azimuth = processing.dwnsampling(zenith, "RGGB"), processing.dwnsampling(azimuth, "RGGB")

    angulartresh = 2

    # ___________________________________________________________________________
    # Pre-allocation of variables
    imtot = np.zeros((int(metambiant["Image ImageLength"].values[0]/4), int(metambiant["Image ImageWidth"].values[0]/2), 3))
    imtot_mask = np.zeros(imtot.shape)
    gain = np.empty(len(image_path))
    exposure = np.empty(len(image_path))

    val = [np.array([]), np.array([]), np.array([])]

    for n, path in enumerate(image_path):
        print("Processing image number {0}".format(n))

        # Reading image
        im_op, met_op = processing.readDNG_insta360(path, which_image=wlens)
        im_op -= ambiant

        # Downsampling
        im_dws = processing.dwnsampling(im_op, "RGGB", ave=True)

        # Storing gain and exposure
        gain[n] = met_op["Image ISOSpeedRatings"].values[0]
        exposure[n] = processing.ratio2float([met_op["Image ExposureTime"].values[0]])[0]

        for i in range(im_dws.shape[2]):
            im = im_dws[:, :, i]
            val[i] = np.append(val[i], im[zenith[:, :, i] < angulartresh])

        # Addition of all image (for visualization)
        imtot += im_dws

    print(exposure)
    print(gain)
    print(val)

    DN_mean = np.array([np.mean(i) for i in val])
    DN_std = np.array([np.std(i) for i in val])

    print(DN_mean)
    print(DN_std)

    calib_coeff = ave_radiance/(DN_mean/(gain[0]*0.01*exposure[0]))

    # Uncertainty
    delta_calib_coeff = (DN_std/DN_mean) * calib_coeff

    # Percentage of difference
    error = abs(100 * (calib_coeff - calib_coeff_sphere)/calib_coeff)


    print("Calibration coefficient red, green, blue:")
    print(calib_coeff)
    #print(delta_calib_coeff)

    print("Calibration coefficient integrating sphere red, green, blue:")
    print(calib_coeff_sphere)
    print("red: {0:.1f} %, green: {1:.1f} %, blue: {2:.1f} %".format(error[0], error[1], error[2]))


    # ___________________________________________________________________________
    # Figures
    # Figure of image tot
    fig1, ax1 = plt.subplots(1, 3)

    ax1[0].imshow(imtot[:, :, 0])
    ax1[0].contour(zenith[:, :, 0] < angulartresh, linewidths=3)
    ax1[0].plot(int(geocal["centerpoint"][0]/2), int(geocal["centerpoint"][1]/2), 'ko', markersize=2,
                markerfacecolor="None", markeredgewidth=1)
    ax1[0].axhline(int(cpoint[1]/2), 0, 3456 / 2)
    ax1[0].axvline(int(cpoint[0]/2), 0, 3456 / 2)

    ax1[1].imshow(imtot[:, :, 1])
    ax1[1].contour(zenith[:, :, 1] < angulartresh, linewidths=3)
    ax1[1].plot(int(geocal["centerpoint"][0]/2), int(geocal["centerpoint"][1]/2), 'ko', markersize=2,
                markerfacecolor="None", markeredgewidth=1)
    ax1[1].axhline(int(cpoint[1]/2), 0, 3456 / 2)
    ax1[1].axvline(int(cpoint[0]/2), 0, 3456 / 2)

    ax1[2].imshow(imtot[:, :, 2])
    ax1[2].contour(zenith[:, :, 0] < angulartresh, linewidths=3)
    ax1[2].plot(int(geocal["centerpoint"][0]/2), int(geocal["centerpoint"][1]/2), 'ko', markersize=2,
                markerfacecolor="None", markeredgewidth=1)
    ax1[2].axhline(int(cpoint[1] / 2), 0, 3456 / 2)
    ax1[2].axvline(int(cpoint[0] / 2), 0, 3456 / 2)

    # Figure of the spectral irradiance
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    ax2.plot(wl_irr, irr_50, label="Irradiance 50 cm ")
    ax2.plot(wl_irr, irr, label="Irradiance {0:.2f} cm".format(distance_lamp))

    ax2.set_xlabel("Wavelength [nm]")
    ax2.set_ylabel("Absolute irradiance [$\mathrm{mW /m^{2} \cdot nm}$]")

    ax2.legend(loc="best")

    # Figure spectral radiance
    fig3, ax3 = plt.subplots(1, 2, figsize=(9, 5))

    sphere_radiance = sphere_source["lum_mean"]
    wl_sphere = sphere_source["wavelength"]
    condition = (400 <= wl_sphere) & (wl_sphere <= 700)

    wl_sphere = wl_sphere[condition]
    sphere_radiance = sphere_radiance[condition]

    ax3[0].plot(wl_irr, radiance, label="Spectralon")
    ax3[0].plot(wl_sphere, sphere_radiance, label="Sphere")

    ax3[0].set_yscale("log")

    ax3[0].set_xlabel("Wavelength [nm]")
    ax3[0].set_ylabel("Absolute radiance [$\mathrm{W /m^{2} \cdot sr \cdot nm}$]")

    ax3[0].legend(loc="best")

    ax3[1].plot(wl_irr, radiance/np.max(radiance), label="Spectralon")
    ax3[1].plot(wl_sphere, sphere_radiance/np.max(sphere_radiance), label="Sphere")

    ax3[1].set_xlabel("Wavelength [nm]")
    ax3[1].set_ylabel("Radiance normalized")

    ax3[1].legend(loc="best")

    fig3.tight_layout()

    # Figure sensor spectrum
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)

    ax4.plot(wl_sensor, qe_norm[:, 0], color="r")
    ax4.plot(wl_sensor, qe_norm[:, 1], color="g")
    ax4.plot(wl_sensor, qe_norm[:, 2], color="b")

    ax4.set_xlabel("Wavelength [nm]")
    ax4.set_ylabel("Sensor relative spectral response")

    # Figure convolution sensor spectral response x spectral radiance
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    ax5.plot(wl_irr, radiance_conv)

    ax5.set_xlabel("Wavelength [nm]")
    ax5.set_ylabel("Radiance x normalized spectral response [$\mathrm{W /m^{2} \cdot sr \cdot nm}$]")

    plt.show()
