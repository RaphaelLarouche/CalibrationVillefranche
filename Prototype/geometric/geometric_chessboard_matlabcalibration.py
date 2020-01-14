# -*- coding: utf-8 -*-
"""
Python script to analyze the geometric data taken with a chessboard target. The script make use of CV2 corner detection
and the third part execute a Matlab script using the Omnidirectional camera calibration toolbox of Scaramuzza et al.

"""

if __name__ == "__main__":

    #  Importation of standard modules
    import numpy as np
    import cv2
    import glob
    import matlab.engine
    import matplotlib.pyplot as plt
    import time
    import os.path
    import scipy.io

    #  Importation of other modules
    import cameracontrol.cameracontrol as cc

    # Functions
    def detect_corners(img):

        # Shape of image
        height, width = img.shape

        # Refinement criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)

        # Resizing for speed optimization
        resizefactor = 1
        img_dwnsca = cv2.resize(img, (int(width / resizefactor), int(height / resizefactor)))  # Resize image

        ret, corners_dwnsca = cv2.findChessboardCorners(img_dwnsca, (8, 6), None)

        print(ret)

        corners_refine = False
        if ret:
            corners = corners_dwnsca
            corners[:, 0][:, 0] = corners_dwnsca[:, 0][:, 0] * resizefactor
            corners[:, 0][:, 1] = corners_dwnsca[:, 0][:, 1] * resizefactor

            # Refinement
            corners_refine = cv2.cornerSubPix(img, np.float32(corners), (5, 5), (-1, -1), criteria)
            corners_refine = corners_refine[:, 0]

            # Visualisation
            cv2.drawChessboardCorners(img, (8, 6), corners_refine, ret)
            cv2.imshow("image", cv2.resize(img, (int(width / 2), int(height / 2))))
            cv2.waitKey(1)

        return corners_refine

    # *** Code beginning ***
    # Creating object from class ProcessImage
    processing = cc.ProcessImage()

    # Files path red, green and blue
    genpath = processing.folder_choice()
    #genpath = "/Volumes/KINGSTON/Quebec/Prototype/Geometric_chess/geometric_proto_air/geometric_proto_air_20191211_2x2_04_ndws"

    images_path = glob.glob(genpath + "/IMG_*.png")
    images_path.sort()

    creation_time = os.path.getmtime(images_path[0])
    date = time.strftime("%Y%m%d_%H%M", time.gmtime(creation_time))
    print(date)

    # ___________________________________________________________________________
    # Corner detection
    point_x = np.array([])
    point_y = np.array([])

    image_tot = np.array([])

    i = 0
    for n, fname in enumerate(images_path):

        # Opening image
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners = detect_corners(img)

        if np.any(corners):
            point_x = np.append(point_x, corners[:, 0])
            point_y = np.append(point_y, corners[:, 1])

            image_tot = np.append(image_tot, img.flatten())

            i += 1

    point_x = point_x.reshape((i, 48)).T
    point_y = point_y.reshape((i, 48)).T

    points = np.empty((48, 2, i))
    points[:, 0, :] = point_x
    points[:, 1, :] = point_y

    image_tot = image_tot.reshape((i, 972, 1296)).T

    matlab_points = matlab.double(points.tolist())
    matlab_images = matlab.double(image_tot.tolist())

    print(matlab_points.size)
    print(matlab_images.size)

    # ___________________________________________________________________________
    # Matlab engine for the geometric calibration

    eng = matlab.engine.start_matlab()

    if "air" in genpath:
        fisheyeParams, Imsize, DistorsionCenter, MapCoeff, theta_rho = eng.calibration_fisheye_python(matlab_images,
                                                                                                      matlab_points,
                                                                                                      date,
                                                                                                      "air",
                                                                                                      nargout=5)

    elif "water" in genpath:
        fisheyeParams, Imsize, DistorsionCenter, MapCoeff, theta_rho = eng.calibration_fisheye_python(matlab_images,
                                                                                                      matlab_points,
                                                                                                      date,
                                                                                                      "water",
                                                                                                      nargout=5)
    else:
        raise ValueError("Not a valid path name...")

    theta_rho = np.asarray(theta_rho)

    # ___________________________________________________________________________
    # Fit radial position from center vs. scene angle
    popt, pcov = processing.geometric_curvefit_forcedzero(theta_rho[:, 0], abs(theta_rho[:, 1]))
    radial_data = np.linspace(0, 400, 1000)

    # ___________________________________________________________________________
    # Figure
    # Figure 1
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    ax1.plot(theta_rho[:, 0], theta_rho[:, 1],  "d", markersize=2, markerfacecolor="None", markeredgecolor='red', markeredgewidth=0.5, label="Data")
    ax1.plot(radial_data, processing.polynomial_fit_forcedzero(radial_data, *popt), 'k-', label="Fit of projection function")

    ax1.set_xlabel("Distance from image center [px]")
    ax1.set_ylabel("Theoretical scene angle [˚]")

    ax1.legend(loc="best")

    plt.show()
