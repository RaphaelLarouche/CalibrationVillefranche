# -*- coding: utf-8 -*-

"""
Python file to analyze data of angular resolution experiment of prototype.
"""

if __name__ == "__main__":

    # Importation of standard modules
    import numpy as np
    import matplotlib.pyplot as plt
    import glob
    import os
    from mpl_toolkits.mplot3d import Axes3D
    import pandas

    # Importation of other modules
    import cameracontrol.cameracontrol as cc

    # ___________________________________________________________________________
    # **** Code beginning  ****

    # Creating object of the class ProcessImage
    processing = cc.ProcessImage()

    # Choice of data with desired binning
    while True:
        answer = input("Which binning do you want to analyze? (4/2): ")
        if answer in ["4", "2"]:
            break

    general_path = "/Volumes/KINGSTON/Villefranche/Prototype/Angular_resolution/"

    # Binning 4x4
    path_to_subfolder = general_path + "resolution_20191129_4x4/"  # Path toward all subfolders
    central_angles = [-100, 0, 100]
    dict_folder = {"resolution_20191129_4x4_01": "$\Delta\\theta$ 1˚",
                   "resolution_20191129_4x4_02": "$\Delta\\theta$ 2˚",
                   "resolution_20191129_4x4_03": "$\Delta\\theta$ 3˚",
                   "resolution_20191129_4x4_04": "$\Delta\\theta$ 4˚"}

    # Binning 2x2
    if answer == "2":

        path_to_subfolder = general_path + "resolution_20191129_2x2/"  # Path toward all subfolders
        delta_angles = [1, 2, 3, 0.5]  # Delta angle at -100, 0 and 100 degrees
        dict_folder = {"resolution_20191129_2x2_01": "$\Delta\\theta$ 1˚",
                       "resolution_20191129_2x2_02": "$\Delta\\theta$ 2˚",
                       "resolution_20191129_2x2_03": "$\Delta\\theta$ 3˚",
                       "resolution_20191129_2x2_half": "$\Delta\\theta$ 0.5˚"}

    subfolders = os.listdir(path_to_subfolder)
    print(subfolders)

    # ___________________________________________________________________________
    # Loop for to process all delta angles

    for subfold in subfolders:
        print(subfold)

        # Image path
        images_path = glob.glob(path_to_subfolder + subfold + "/IMG_*.tif")
        images_path_dark = glob.glob(path_to_subfolder + subfold + "/DARK_*.tif")

        if answer == "2":
            images_path = glob.glob(path_to_subfolder + subfold + "/Filter" + "/IMG_*.tif")
            images_path_dark = glob.glob(path_to_subfolder + subfold + "/Filter" + "/DARK_*.tif")

        # Sorting image path
        images_path.sort()
        images_path_dark.sort()

        # Opening dark image
        imdark, metdark = processing.readTIFF_xiMU(images_path_dark[0])

        # Pre-allocation
        imtot = np.zeros((int(metdark["height"]/2), int(metdark["width"]/2), 3))
        central_centroid = np.empty((len(images_path), 3), dtype=[("y", "float32"), ("x", "float32")])
        central_centroid.fill(np.nan)

        # x and y coordinates
        ximsize, yimsize = imtot.shape[0:2]
        xcoord, ycoord = np.meshgrid(np.arange(0, int(ximsize)), np.arange(0, int(yimsize)))

        # Pre-allocation of figures
        fig1, axes1 = plt.subplots(1, 3)
        fig2, axes2 = plt.subplots(3, 3, figsize=(12, 7))
        fig2.suptitle(dict_folder[subfold])

        axposition = np.arange(0, 9)
        axposition = axposition.reshape(3, 3)

        for i in range(3):
            for j in range(3):
                axes2[i, j].remove()
                axes2[i, j] = fig2.add_subplot(3, 3, axposition[i, j]+1, projection="3d")

        # Loop to process image in each subdirectories
        for n, path in enumerate(images_path):
            print("Processing image number {0}".format(n))

            # Reading images
            im_op, met_op = processing.readTIFF_xiMU(path)
            im_op -= imdark

            # Downsampling
            im_dws = processing.dwnsampling(im_op, "BGGR")

            # Addition of all image
            imtot += im_dws

            # Centroids
            for j in range(im_dws.shape[2]):
                _, regionprops = processing.regionproperties(im_dws[:, :, j], 80)

                if regionprops:
                    yc, xc = regionprops[0].centroid
                    central_centroid[n, j] = regionprops[0].centroid
                    axes1[j].plot(xc, yc, "r+", markersize=3)

        print(central_centroid)

        # Clipping total image
        imtot = np.clip(imtot, 0, 2**12)

        # Plotting image total
        subplot_title_list = ["Red channel: ", "Green channel: ", "Blue channel: "]
        row = 1
        for a, ca in enumerate(central_angles):
            for j in range(imtot.shape[2]):

                ycentr, xcentr = central_centroid[row, j]

                print(ycentr, xcentr)

                xcentr = int(xcentr)
                ycentr = int(ycentr)

                Xcoord = xcoord[ycentr-10:ycentr+10:1, xcentr-10:xcentr+10:1]
                Ycoord = ycoord[ycentr-10:ycentr+10:1, xcentr-10:xcentr+10:1]
                Zdata = imtot[ycentr-10:ycentr+10:1, xcentr-10:xcentr+10:1, j]

                print(Xcoord.shape, Ycoord.shape, Zdata.shape)

                axes1[j].imshow(imtot[:, :, j])

                axes2[a, j].plot_surface(Xcoord, Ycoord, Zdata, cmap="viridis")
                axes2[a, j].set_title(subplot_title_list[j] + str(ca) + "˚")

            row += 3

        # Figure parameters

    plt.show()
