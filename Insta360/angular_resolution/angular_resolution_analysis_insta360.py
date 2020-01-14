# -*- coding: utf-8 -*-

"""
Python file to analyze data of angular resolution experiment of insta360 ONE.
"""

if __name__ == "__main__":

    # Importation of standard modules
    import numpy as np
    import matplotlib.pyplot as plt
    import glob
    from mpl_toolkits.mplot3d import Axes3D
    import os

    # Importation of other modules
    import cameracontrol.cameracontrol as cc

    # Function
    def loop_imagetotal(processing, lens, image_list, imdark, metdark):
        """

        :param processing:
        :param lens:
        :param image_list:
        :param imdark:
        :param metdark:
        :return:
        """

        # Pre-allocation of variables
        imtotal = np.zeros((int(metdark["Image ImageLength"].values[0] / 4),
                            int(metdark["Image ImageWidth"].values[0] / 2), 3))

        centroids = np.empty((len(image_list), 3), dtype=[("y", "float32"), ("x", "float32")])
        centroids.fill(np.nan)

        # Initialization of figure
        fig1, ax1 = plt.subplots(1, 3)

        # Loop
        for n, path in enumerate(image_list):
            print("Processing image number {0}".format(n))

            # Reading data
            im_op, metadata = processing.readDNG_insta360(path, which_image=lens)
            im_op -= imdark

            # Downsampling
            im_dws = processing.dwnsampling(im_op, "RGGB", ave=True)

            # Addition of image
            imtotal += im_dws

            for j in range(im_dws.shape[2]):
                _, regionprops = processing.regionproperties(im_dws[:, :, j], 1000)

                if regionprops:
                    centroids[n, j] = regionprops[0].centroid
                    yc, xc = regionprops[0].centroid
                    ax1[j].plot(xc, yc, "r+", markersize=3)

        imtotal = np.clip(imtotal, 0, 2**14)

        list_title = ["Red channel", "Green channel", "Blue channel"]
        for a, ax in enumerate(ax1):
            ax.imshow(imtotal[:, :, a])
            ax.set_title(list_title[a])

        return imtotal, centroids

    def resolution_figures(centroids, imtotal, suptitle):

        # Random variables
        angle = [-90, 0, 90]
        nb_pixels = 15
        subplot_title_list = ["Red channel: ", "Green channel: ", "Blue channel: "]

        ximsize, yimsize = imtotal.shape[0:2]
        xcoord, ycoord = np.meshgrid(np.arange(0, int(ximsize)), np.arange(0, int(yimsize)))

        # Position of first central spot
        print(centroids.shape[0])
        incr = int(round(centroids.shape[0]/3))
        first_row = int(round(incr/2))

        # Initialization of image
        fig2, axes2 = plt.subplots(3, 3, figsize=(12, 7))
        fig2.suptitle(suptitle)
        axe_position = np.arange(1, 10).reshape(3, 3)

        for i in range(3):
            for j in range(3):
                axes2[i, j].remove()
                axes2[i, j] = fig2.add_subplot(3, 3, axe_position[i, j], projection="3d")

        for a, ca in enumerate(angle):
            for j in range(imtotal.shape[2]):
                ycentr, xcentr = centroids[first_row, j]

                xcentr = int(xcentr)
                ycentr = int(ycentr)

                Xcoord = xcoord[ycentr - nb_pixels:ycentr + nb_pixels:1, xcentr - nb_pixels:xcentr + nb_pixels:1]
                Ycoord = ycoord[ycentr - nb_pixels:ycentr + nb_pixels:1, xcentr - nb_pixels:xcentr + nb_pixels:1]
                Zdata = imtotal[ycentr - nb_pixels:ycentr + nb_pixels:1, xcentr - nb_pixels:xcentr + nb_pixels:1, j]

                axes2[a, j].plot_surface(Xcoord, Ycoord, Zdata, cmap="viridis")
                axes2[a, j].set_title(subplot_title_list[j] + str(ca) + "˚")

            first_row += incr


    # ___________________________________________________________________________
    # **** Code beginning ****

    # Creating object of class ProcessImage
    processing = cc.ProcessImage()

    # Choice of lens and setting the path to data
    # Input of lens to analyzed
    while True:
        answer = input("Which lens do you want to analyze? (c/f): ")
        if answer.lower() in ["c", "f"]:
            break

    general_path = "/Volumes/KINGSTON/Villefranche/Insta360/Angular_resolution/"
    if answer.lower() == "c":

        # Variable to open image
        wlens = "close"

        # Path to data
        impath = general_path + "LensClose/resolution_20191129_1ov2"

    elif answer.lower() == "f":

        # Variable to open image
        wlens = "far"

        # Path to data
        impath = general_path + "LensFar/resolution_20191129_1ov2"

    images_path = glob.glob(impath + "/IMG_*.dng")
    images_path_dark = glob.glob(impath + "/DARK_*.dng")

    images_path.sort()
    images_path_dark.sort()

    print(images_path)

    # Opening dark image
    imdark, metdark = processing.readDNG_insta360(images_path_dark[0], which_image=wlens)

    # ___________________________________________________________________________
    # Loop to treat all image at different delta angle

    # 0.25 degrees

    imtot_025deg, centr_025deg = loop_imagetotal(processing, wlens, images_path, imdark, metdark)

    resolution_figures(centr_025deg, imtot_025deg, "$\Delta \\theta = $ 0.25˚")

    # 0.50 degrees
    images_050deg_90 = images_path[0:9:2]
    images_050deg_0 = images_path[9:18:2]
    images_050deg_pos90 = images_path[18:27:2]

    if answer.lower() == "c":

        images_050deg_90 = images_path[0:9:2]
        images_050deg_0 = images_path[9:18:2]
        images_050deg_pos90 = images_path[18:23:2] + images_path[23:26:2]


    images_050deg = images_050deg_90 + images_050deg_0 + images_050deg_pos90
    imtot_050deg, centr_50deg = loop_imagetotal(processing, wlens, images_050deg, imdark, metdark)

    resolution_figures(centr_50deg, imtot_050deg, "$\Delta \\theta = $ 0.50˚")

    plt.show()
