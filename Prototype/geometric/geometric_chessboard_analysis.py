# -*- coding: utf-8 -*-
"""

"""

if __name__ == "__main__":

    # Importation of standard modules
    import numpy as np
    import glob
    import matplotlib.pyplot as plt
    import scipy.io
    import cv2

    # Importation of other modules
    import cameracontrol.cameracontrol as cc

    # Function

    # *** Code beginning ***
    processing = cc.ProcessImage()

    # Importation of matlab calibration files
    #path_air = "matlab_fisheye_calibrationfiles_cb_air/fisheyeParams_20191211_2152.mat" # After moving CMOS !!!
    path_air = "matlab_fisheye_calibrationfiles_cb_air/fisheyeParams_20191211_1714.mat"  # Before moving CMOS !!!
    path_water = "matlab_fisheye_calibrationfiles_cb_water/fisheyeParams_20191212_1721.mat"

    fisheyeParams_air = scipy.io.loadmat(path_air)
    fisheyeParams_water = scipy.io.loadmat(path_water)

    # Extracting data
    fisheyeParams_air = fisheyeParams_air["fisheyeParams"]
    fisheyeParams_water = fisheyeParams_water["fisheyeParams"]

    intrinsics_air = fisheyeParams_air["Intrinsics"][0][0]
    intrinsics_water = fisheyeParams_water["Intrinsics"][0][0]

    print(intrinsics_air)

    distortion_center_a = intrinsics_air["DistortionCenter"][0][0]
    distortion_center_w = intrinsics_water["DistortionCenter"][0][0]

    print(distortion_center_a.shape)

    imsize_a = intrinsics_air["ImageSize"][0][0]
    imsize_w = intrinsics_water["ImageSize"][0][0]

    theta_rho_a = fisheyeParams_air["thetaAndrho"][0][0]
    theta_rho_w = fisheyeParams_water["thetaAndrho"][0][0]

    # Path to .png image
    gen_path_img = "/Volumes/KINGSTON/Quebec/Prototype/Geometric_chess/"
    path_img_air = gen_path_img + "geometric_proto_air/geometric_proto_air_20191211_2x2_03_ndws"  # correspond to fisheyeParams_20191211_1714.mat
    path_img_water = gen_path_img + "geometric_proto_water/geometric_proto_20191212_2x2_06"

    images_path_a = glob.glob(path_img_air + "/IMG_*.png")
    images_path_w = glob.glob(path_img_water + "/IMG_*.png")

    images_path_a.sort()
    images_path_w.sort()

    # ___________________________________________________________________________
    # Fitting theta vs. radial position
    popt_a, pcov_a = processing.geometric_curvefit_forcedzero(theta_rho_a[:, 0], abs(theta_rho_a[:, 1]))
    popt_w, pcov_w = processing.geometric_curvefit_forcedzero(theta_rho_w[:, 0], abs(theta_rho_w[:, 1]))

    radial_data = np.linspace(0, 400, 1000)

    # ___________________________________________________________________________
    # Fitting limits of Field of view

    figfov, axfov = plt.subplots(1, 2)

    estimate_limit_rad = [312, 342]

    FOVair = processing.polynomial_fit_forcedzero(estimate_limit_rad[0], *popt_a)
    FOVwater = processing.polynomial_fit_forcedzero(estimate_limit_rad[1], *popt_w)

    print("Fied of view limits air {0:.2f}\nFied of view limits water {1:.2f}".format(FOVair, FOVwater))

    titles = ["Air", "Water"]
    dist_cent = [distortion_center_a, distortion_center_w]

    plt.ion()
    for n, imname in enumerate(zip(images_path_a, images_path_w)):

        for i, im in enumerate(imname):

            img = cv2.imread(im)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, thres = cv2.threshold(img, 5, 255, cv2.THRESH_TOZERO)

            center_coordinates = (int(round(dist_cent[i][0, 0])), int(round(dist_cent[i][0, 1])))

            thresh = cv2.circle(thres, center_coordinates, estimate_limit_rad[i], (255, 0, 0), 1)

            axfov[i].imshow(thres, "gray")
            axfov[i].set_title(titles[i])

        plt.pause(1)

    plt.ioff()

    # ___________________________________________________________________________
    # Figures
    #  Data and fit figure
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    ax1.plot(theta_rho_a[:, 0], abs(theta_rho_a[:, 1]), 'o')
    ax1.plot(theta_rho_w[:, 0], abs(theta_rho_w[:, 1]), 'o')
    ax1.plot(radial_data, processing.polynomial_fit_forcedzero(radial_data, *popt_a))
    ax1.plot(radial_data, processing.polynomial_fit_forcedzero(radial_data, *popt_w))

    ax1.set_xlim([0, 400])

    ax1.set_xlabel("Radial position from center [px]")
    ax1.set_ylabel("Scene angle [˚]")

    # Only fit figure
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    ax2.plot(radial_data, processing.polynomial_fit_forcedzero(radial_data, *popt_a), color="#1f77b4", linewidth=1.5, label="Chessboard calibration air")
    ax2.plot(radial_data, processing.polynomial_fit_forcedzero(radial_data, *popt_w), color="#ff7f0e", linewidth=1.5, label="Chessboard calibration water")
    #ax2.plot(radial_data, processing.polynomial_fit_forcedzero(radial_data, *popt_a)*1/1.33, 'r-.')

    ax2.axvline(estimate_limit_rad[0], 0, 130, linestyle="--", color="#1f77b4")
    ax2.axvline(estimate_limit_rad[1], 0, 130, linestyle="--", color="#ff7f0e")
    ax2.axhline(FOVair, 0, 400, linestyle="--", color="#1f77b4")
    ax2.axhline(FOVwater, 0, 400, linestyle="--", color="#ff7f0e")

    ax2.text(40, FOVair + 2, "{:.2f} FOV".format(FOVair))
    ax2.text(40, FOVwater + 2, "{:.2f} FOV".format(FOVwater))

    ax2.set_xlim([0, 400])

    ax2.set_xlabel("Radial position from image center [px]")
    ax2.set_ylabel("Scene angle [˚]")

    ax2.legend(loc="best")


    # ___________________________________________________________________________
    # Saving geometric calibration data
    while True:
        inputsav = input("Do you want to save the calibration results? (y/n) : ")
        inputsav = inputsav.lower()
        if inputsav in ["y", "n"]:
            break

    if inputsav == "y":
        # Air
        name_a = "geo_calibration_2x2_air_" + path_air[-17:-4]
        savename_a = "geometric_calibrationfiles_cb_air/" + name_a + ".npz"

        np.savez(savename_a, imagesize=imsize_a, centerpoint=distortion_center_a, fitparams=popt_a)

        # Water
        name_w = "geo_calibration_2x2_water_" + path_water[-17:-4]
        savename_w = "geometric_calibrationfiles_cb_water/" + name_w + ".npz"

        np.savez(savename_w, imagesize=imsize_w, centerpoint=distortion_center_w, fitparams=popt_w)


    plt.show()
