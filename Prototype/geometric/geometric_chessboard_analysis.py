# -*- coding: utf-8 -*-
"""
Analysis of results of geometric calibration of the PROTOTYPE using a chessboard target.
"""

# Importation of standard modules
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.io
import cv2
import json
from matplotlib import rc

# Importation of other modules
import cameracontrol.cameracontrol as cc

# Function
def projection(radial_distance, projectiontype="equisolid"):
    if projectiontype == "equisolid":
        angle = 2 * np.arcsin(radial_distance/(2*173))

    return angle * 180 / np.pi


def projection_errors(fisheyeParams):
    """
    Function to outputted the radial distance of chessboard points (relative to center) detected and  reprojected.

    :param fisheyeParams: Data which is a Matlab Structure containing all paramters related to the fisheye calibration.
    :return: (radial_distance - reprojection points, radial_distance_fitted_points - corner detection algorithm)
    """

    # Finding radial distance from center of each reprojected points
    reprojection_points = fisheyeParams["ReprojectedPoints"][0][0]
    intrinsics = fisheyeParams["Intrinsics"][0][0]
    xcenter, ycenter = intrinsics["DistortionCenter"][0][0][0, 0], intrinsics["DistortionCenter"][0][0][0, 1]

    radial_distance = np.sqrt((reprojection_points[:, 0, :] - xcenter)**2 + (reprojection_points[:, 1, :] - ycenter)**2)

    # Reprojection mean error per image (eucledian distance between corners detected and reprojections)
    reprojection_error = fisheyeParams["ReprojectionErrors"][0][0]
    reprojection_error_x = reprojection_error[:, 0, :]
    reprojection_error_y = reprojection_error[:, 1, :]

    eucledian_error = np.sqrt(reprojection_error_x**2 + reprojection_error_y**2)

    mean_x = np.mean(reprojection_error_x, axis=0)
    mean_y = np.mean(reprojection_error_y, axis=0)
    mean_e = np.mean(eucledian_error, axis=0)

    fitted_points = reprojection_points + reprojection_error
    radial_distance_fitted_points = np.sqrt((fitted_points[:, 0, :] - xcenter)**2 + (fitted_points[:, 1, :] - ycenter)**2)

    return radial_distance, radial_distance_fitted_points, mean_x, mean_y, mean_e, eucledian_error


if __name__ == "__main__":

    # *** Code beginning ***
    processing = cc.ProcessImage()

    # Importation of matlab calibration files
    #path_air = "matlab_fisheye_calibrationfiles_cb_air/fisheyeParams_20191211_2152.mat" # After moving CMOS !!!
    #path_air = "matlab_fisheye_calibrationfiles_cb_air/fisheyeParams_20191211_1714.mat"  # Before moving CMOS !!!
    #path_air = "matlab_fisheye_calibrationfiles_cb_air/fisheyeParams_20200218_2304.mat"
    #path_air = "matlab_fisheye_calibrationfiles_cb_air/fisheyeParams_20200220_1908.mat"
    path_air = "matlab_fisheye_calibrationfiles_cb_air/fisheyeParams_20200306_1611.mat"
    #path_water = "matlab_fisheye_calibrationfiles_cb_water/fisheyeParams_20201212_1721.mat"
    path_water = "matlab_fisheye_calibrationfiles_cb_water/fisheyeParams_20200225_2052.mat"


    fisheyeParams_air = scipy.io.loadmat(path_air)
    fisheyeParams_water = scipy.io.loadmat(path_water)

    # Extracting data

    # Fisheye parameters
    fisheyeParams_air = fisheyeParams_air["fisheyeParams"]
    fisheyeParams_water = fisheyeParams_water["fisheyeParams"]

    # Intrinsics
    intrinsics_air = fisheyeParams_air["Intrinsics"][0][0]
    intrinsics_water = fisheyeParams_water["Intrinsics"][0][0]

    print(intrinsics_air)

    distortion_center_a = intrinsics_air["DistortionCenter"][0][0]
    print(distortion_center_a)
    distortion_center_w = intrinsics_water["DistortionCenter"][0][0]

    print(distortion_center_a.shape)

    imsize_a = intrinsics_air["ImageSize"][0][0]
    imsize_w = intrinsics_water["ImageSize"][0][0]

    theta_rho_a = fisheyeParams_air["thetaAndrho"][0][0]
    theta_rho_w = fisheyeParams_water["thetaAndrho"][0][0]

    # Path to .png image
    gen_path_img = "/Volumes/KINGSTON/Quebec/Prototype/Geometric_chess/"
    #path_img_air = gen_path_img + "geometric_proto_air/geometric_proto_air_20191211_2x2_03_ndws"  # correspond to fisheyeParams_20191211_1714.mat
    #path_img_air = gen_path_img + "geometric_proto_air/geometric_proto_air_20200218/20200218_2x2_ndws_01"
    path_img_air = gen_path_img + "geometric_proto_air/geometric_proto_air_20200220/20200220_2x2_ndws_01"
    #path_img_water = gen_path_img + "geometric_proto_water/geometric_proto_20191212_2x2_06"
    path_img_water = gen_path_img + "geometric_proto_water/geometric_proto_20200225_2x2_04"

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
    # Derivative of projection functions
    derivative_air = np.gradient(processing.polynomial_fit_forcedzero(radial_data, *popt_a), radial_data)
    derivative_water = np.gradient(processing.polynomial_fit_forcedzero(radial_data, *popt_w), radial_data)

    # ___________________________________________________________________________
    # Reprojection errors
    rd_air, rd_air_fitted, _, _, mean_e_air, e_air = projection_errors(fisheyeParams_air)
    rd_water, rd_water_fitted, _, _, mean_e_water, e_water = projection_errors(fisheyeParams_water)

    # Average reprojection error
    total_error_air = np.mean(mean_e_air)
    total_error_water = np.mean(mean_e_water)

    # Interpolation of gradient at each position
    grad_air = np.interp(rd_air, radial_data, derivative_air)
    grad_water = np.interp(rd_water, radial_data, derivative_water)

    # Angular error according to gradient
    error_grad_air = grad_air * e_air
    error_grad_water = grad_water * e_water

    # Angular error according to radial distance of fitted points vs. radial distance of reprojected points
    error_deg_air = abs(processing.polynomial_fit_forcedzero(rd_air_fitted, *popt_a) - processing.polynomial_fit_forcedzero(rd_air, *popt_a))
    error_deg_water = abs(processing.polynomial_fit_forcedzero(rd_water_fitted, *popt_w) - processing.polynomial_fit_forcedzero(rd_water, *popt_w))

    # MOVING AVERAGE !
    error_deg_air = error_deg_air.ravel()
    error_deg_water = error_deg_water.ravel()

    i_rd_air_sorted = np.argsort(rd_air.ravel())
    rd_air_sorted = rd_air.ravel()[i_rd_air_sorted]
    error_deg_air = error_deg_air[i_rd_air_sorted]

    i_rd_water_sorted = np.argsort(rd_water.ravel())
    rd_water_sorted = rd_water.ravel()[i_rd_water_sorted]
    error_deg_water = error_deg_water[i_rd_water_sorted]

    N = 30
    error_deg_air_movingavg = np.convolve(error_deg_air, np.ones((N,))/N, mode='valid')
    error_deg_water_movingavg = np.convolve(error_deg_water, np.ones((N,))/N, mode='valid')

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

        #plt.pause(1)

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
    ax1.plot(radial_data, projection(radial_data))

    ax1.set_xlim([0, 400])

    ax1.set_xlabel("Radial position from center [px]")
    ax1.set_ylabel("Scene angle [˚]")

    # Only fit figure
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    ax2.plot(radial_data, processing.polynomial_fit_forcedzero(radial_data, *popt_a), color="#1f77b4", linewidth=1.5, label="Air calibration")
    ax2.plot(radial_data, processing.polynomial_fit_forcedzero(radial_data, *popt_a) * 1/1.33, color="k", linewidth=1.5)
    ax2.plot(radial_data, processing.polynomial_fit_forcedzero(radial_data, *popt_w), color="#ff7f0e", linewidth=1.5, label="Water calibration")
    #ax2.plot(radial_data, processing.polynomial_fit_forcedzero(radial_data, *popt_a)*1/1.33, 'r-.')

    ax2.axvline(estimate_limit_rad[0], 0, 130, linestyle="--", color="#1f77b4")
    ax2.axvline(estimate_limit_rad[1], 0, 130, linestyle="--", color="#ff7f0e")
    ax2.axhline(FOVair, 0, 400, linestyle="--", color="#1f77b4")
    ax2.axhline(FOVwater, 0, 400, linestyle="--", color="#ff7f0e")

    ax2.text(40, FOVair + 2, "{:.2f} FOV".format(FOVair))
    ax2.text(40, FOVwater + 2, "{:.2f} FOV".format(FOVwater))

    ax2.set_xlim([0, 350])

    ax2.set_xlabel("Radial position from image center [px]")
    ax2.set_ylabel("Scene angle [˚]")

    ax2.legend(loc="best")

    # Reprojection error in function of radial position

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)

    colorb = 'tab:blue'
    ax3.scatter(rd_air, e_air, color=colorb, label="air average error {:.3f} px".format(total_error_air))
    ax3.set_xlabel("Distance from center [px]")
    ax3.set_ylabel("Reprojection error [px]", color=colorb)
    ax3.tick_params(axis='y', labelcolor=colorb)
    ax3.legend(loc="best")

    colorr = 'tab:red'
    ax3_twin = ax3.twinx()
    ax3_twin.scatter(rd_air, error_grad_air, color=colorr)
    ax3_twin.set_ylabel("Reprojection error [˚]", color=colorr)
    ax3_twin.tick_params(axis='y', labelcolor=colorr)

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)

    ax4.scatter(rd_water, e_water, color=colorb, label="water average error {:.3f} px".format(total_error_water))
    ax4.set_xlabel("Distance from center [px]")
    ax4.set_ylabel("Reprojection error [px]", color=colorb)
    ax4.tick_params(axis="y", labelcolor=colorb)
    ax4.legend(loc="best")

    ax4_twin = ax4.twinx()
    ax4_twin.scatter(rd_water, error_grad_water, color=colorr)
    ax4_twin.set_ylabel("Reprojection error [˚]", color=colorr)
    ax4_twin.tick_params(axis='y', labelcolor=colorr)

    # Gradient
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)

    ax5.plot(radial_data, derivative_air, label="Air")
    ax5.plot(radial_data, derivative_water, label="Water")

    ax5.set_xlim([0, 350])

    ax5.set_xlabel("Distance from center [px]")
    ax5.set_ylabel(r"Gradient $d{\theta}/dr$ [˚/px]")

    ax5.legend(loc="best")

    # Error from radial distance between fitted points and reprojected points
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(111)

    ax6.scatter(rd_air_sorted, error_deg_air, s=4, label="Angular error for target in air")
    ax6.plot(rd_air_sorted[N//2-1:-N//2], error_deg_air_movingavg, "r", alpha=0.8, linewidth=3, label="Moving average using {} values".format(N))
    ax6.set_yscale("log")
    ax6.set_ylim([0.0001, 10])

    ax6.set_xlabel("Distance from center [px]")
    ax6.set_ylabel("Reprojection error [˚]")

    ax6.legend(loc="best")

    fig7 = plt.figure()
    ax7 = fig7.add_subplot(111)

    ax7.scatter(rd_water_sorted, error_deg_water, s=4, label="Angular error for target in water")
    ax7.plot(rd_water_sorted[N//2-1:-N//2], error_deg_water_movingavg, "r", alpha=0.8, linewidth=3, label="Moving average using {} values".format(N))
    ax7.set_yscale("log")
    ax7.set_ylim([0.0001, 10])

    ax7.set_xlabel("Distance from center [px]")
    ax7.set_ylabel("Reprojection error [˚]")

    ax7.legend(loc="best")

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

        np.savez(savename_a, imagesize=np.squeeze(imsize_a), centerpoint=np.squeeze(distortion_center_a),
                 fitparams=np.squeeze(popt_a))

        # Water
        name_w = "geo_calibration_2x2_water_" + path_water[-17:-4]
        savename_w = "geometric_calibrationfiles_cb_water/" + name_w + ".npz"

        np.savez(savename_w, imagesize=np.squeeze(imsize_w), centerpoint=np.squeeze(distortion_center_w),
                 fitparams=np.squeeze(popt_w))

    plt.show()
