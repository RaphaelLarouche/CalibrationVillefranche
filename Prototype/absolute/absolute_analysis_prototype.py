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

    # Importation of other modules
    import cameracontrol.cameracontrol as cc

    # *** Code beginning ***
    # Creating object from class ProcessImage
    processing = cc.ProcessImage()

    # Choice of data with desired binning
    while True:
        answer = input("Which binning do you want to analyze? (4/2): ")
        if answer in ["4", "2"]:
            break

    # General path
    genpath = "~/PycharmProjects/CalibrationVillefranche/"
    genpath_kingston = "/Volumes/KINGSTON/Villefranche/"

    # binning 4x4
    impath = genpath_kingston + "Prototype/Absolute/absolute_proto_20191114_4x4/absolute_proto_20191114_4x4_15000us"

    # Open geometric calibration
    geocalib = np.load(os.path.dirname(os.getcwd()) + "/geometric/geometric_calibrationfiles_air/geo_calibration_results_4x4_20191115.npz")

    if answer == "2":

        # binning 2x2 -- uniquement 4x4 pour le moment
        impath = genpath_kingston + "Prototype/Absolute/absolute_proto_20191128_2x2/absolute_proto_20191128_2x2_17000us"

        # Open geometric calibration
        geocalib = np.load(os.path.dirname(os.getcwd()) + "/geometric/geometric_calibrationfiles_air/geo_calibration_results_2x2_20191115.npz")

    images_path = glob.glob(impath + "/IMG_*tif")
    ambiance_path = glob.glob(impath + "/DARK_*.tif")

    images_path.sort()
    ambiance_path.sort()

    # Opening ambiance image
    ambiant, metambiant = processing.readTIFF_xiMU(ambiance_path[0])

    print(metambiant)

    # ___________________________________________________________________________
    # Angular coordinates of each pixel
    zenith, azimuth = processing.angularcoordinates(geocalib["imagesize"], geocalib["centerpoint"], geocalib["fitparams"])

    zenith = processing.dwnsampling(zenith, "BGGR")
    azimuth = processing.dwnsampling(azimuth, "BGGR")

    angular_thresh = 5

    # ___________________________________________________________________________
    # Mean radiance in each channel

    # Opening spectral response of camera
    sensor_rsr_data = pandas.read_csv(genpath + "cameracontrol/MT9P031_RSR/MT9P031.csv", sep=";")
    sensor_rsr_data["R"] = sensor_rsr_data["R"] / np.nanmax(sensor_rsr_data["R"])
    sensor_rsr_data["G"] = sensor_rsr_data["G"] / np.nanmax(sensor_rsr_data["G"])
    sensor_rsr_data["B"] = sensor_rsr_data["B"] / np.nanmax(sensor_rsr_data["B"])

    sensor_rsr_data = sensor_rsr_data.dropna()
    print(sensor_rsr_data)

    # Opening spectral irradiance of spectralon plate
    irr_data = pandas.read_excel(genpath + "files/FEL_GS_1015_opt.xlsx")
    irr_data = irr_data.where((400 <= irr_data["wavelength [nm]"]) & (irr_data["wavelength [nm]"] <= 850))
    irr_data = irr_data.dropna()

    reflectance_data = pandas.read_csv(genpath + "files/spectralon.txt", delimiter="\t")
    reflectance_data = reflectance_data.where((400 <= reflectance_data["Wl"]) & (reflectance_data["Wl"] <= 850))
    reflectance_data = reflectance_data.dropna()
    reflectance_data = reflectance_data[::10]

    # Calcul of irradiance at 179.9 cm
    lamp_distance = 179.9

    irradiance = irr_data["absolute irradiance [mW m-2 nm-1]"] * (50/lamp_distance)**2

    # Calcul of radiance
    radiance = (irradiance * np.array(reflectance_data["R"]) * 0.001)/np.pi  # W m-2 sr-1 nm-1

    # Interpolation or sensor relative spectral response
    print(radiance.shape)

    sensor_rsr_inter = np.empty((radiance.shape[0], 3))
    sensor_rsr_inter[:, 0] = processing.interpolation(irr_data["wavelength [nm]"], (sensor_rsr_data["RW"], sensor_rsr_data["R"]))
    sensor_rsr_inter[:, 1] = processing.interpolation(irr_data["wavelength [nm]"], (sensor_rsr_data["GW"], sensor_rsr_data["G"]))
    sensor_rsr_inter[:, 2] = processing.interpolation(irr_data["wavelength [nm]"], (sensor_rsr_data["BW"], sensor_rsr_data["B"]))

    # Bandpass
    bwr = np.trapz(sensor_rsr_inter[:, 0], x=irr_data["wavelength [nm]"])
    bwg = np.trapz(sensor_rsr_inter[:, 1], x=irr_data["wavelength [nm]"])
    bwb = np.trapz(sensor_rsr_inter[:, 2], x=irr_data["wavelength [nm]"])

    # Convoluted radiance
    radiance_conv = sensor_rsr_inter * np.tile(np.array(radiance), (3, 1)).T

    # Integration for average radiance in each band
    intr = np.trapz(radiance_conv[:, 0], x=irr_data["wavelength [nm]"]) / bwr
    intg = np.trapz(radiance_conv[:, 1], x=irr_data["wavelength [nm]"]) / bwg
    intb = np.trapz(radiance_conv[:, 2], x=irr_data["wavelength [nm]"]) / bwb

    average_radiance = np.array([intr, intg, intb])
    print(average_radiance)

    # ___________________________________________________________________________
    # Loop to get the mean DN inside the region of interest
    imtot = np.zeros((int(metambiant["height"]/2), int(metambiant["width"]/2), 3))
    imtot_mask = np.zeros(imtot.shape)
    gain = np.empty(len(images_path))
    exposure = np.empty(len(images_path))

    DNval = [np.array([]), np.array([]), np.array([])]

    for n, path in enumerate(images_path):
        print("Processing image nubmer {0}".format(n))

        # Reading image
        im_op, met_op = processing.readTIFF_xiMU(path)
        im_op -= ambiant

        # Downsamplig
        im_dws = processing.dwnsampling(im_op, "BGGR")

        # Storing gain and exposure
        gain[n] = processing.gain_linear(met_op["gain_db"])
        exposure[n] = processing.exposure_second(met_op["exposure_time_us"])

        for i in range(im_dws.shape[2]):
            im = im_dws[:, :, i]
            DNval[i] = np.append(DNval[i], im[zenith[:, :, i] < angular_thresh])

        # Addition of all image
        imtot += im_dws

    print(exposure)
    print(gain)
    print(DNval)

    # Computation of the mean of the digital numbers and the standard deviation

    DN_mean = np.array([np.mean(i) for i in DNval])
    DN_std = np.array([np.std(i) for i in DNval])

    # Relative standard deviation
    percentage = (DN_std / DN_mean) * 100
    channel = ["Red", "Green", "Blue"]

    # Calibration coefficient
    calibration_coeff = average_radiance / (DN_mean / (gain[0] * exposure[0]))

    print(DN_mean)
    print(DN_std)
    print(calibration_coeff)

    for i in range(DN_mean.shape[0]):

        print("{0} : {1:.2f} {2:.2f} ({3:.2f} %)".format(channel[i], DN_mean[i], DN_std[i], percentage[i]))
        print("{0} : {1:.7E}".format("Coefficient " + channel[i], calibration_coeff[i]))

    # ___________________________________________________________________________
    # Figures

    # Figure spectral response and interpolation
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    ax1.plot(sensor_rsr_data["RW"], sensor_rsr_data["R"], 'r')
    ax1.plot(sensor_rsr_data["GW"], sensor_rsr_data["G"], 'g')
    ax1.plot(sensor_rsr_data["BW"], sensor_rsr_data["B"], 'b')

    ax1.plot(irr_data["wavelength [nm]"], sensor_rsr_inter[:, 0], 'ro')
    ax1.plot(irr_data["wavelength [nm]"], sensor_rsr_inter[:, 1], 'go')
    ax1.plot(irr_data["wavelength [nm]"], sensor_rsr_inter[:, 2], 'bo')

    # Figure imagetot
    fig2, ax2 = plt.subplots(1, 3)

    ax2[0].imshow(imtot[:, :, 0])
    ax2[0].contour(zenith[:, :, 0] < angular_thresh, linewidths=2)

    ax2[1].imshow(imtot[:, :, 1])
    ax2[1].contour(zenith[:, :, 1] < angular_thresh, linewidths=2)

    ax2[2].imshow(imtot[:, :, 2])
    ax2[2].contour(zenith[:, :, 2] < angular_thresh, linewidths=2)

    plt.show()
