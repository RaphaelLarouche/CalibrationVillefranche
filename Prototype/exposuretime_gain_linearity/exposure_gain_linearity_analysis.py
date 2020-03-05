# -*- coding: utf-8 -*-
"""
Script to analyze exposure time and gain linearity.
"""

if __name__ == "__main__":

    # Importation of standard modules
    import numpy as np
    import matplotlib.pyplot as plt
    import glob

    # Importation of other modules
    import cameracontrol.cameracontrol as cameracontrol

    # Functions
    def circle_mask(im, xcenter, ycenter, numberpixel):
        """

        :param im:
        :param xcenter:
        :param ycenter:
        :return:
        """
        size = im.shape
        xcoord, ycoord = np.meshgrid(np.arange(0, size[1]), np.arange(0, size[0]))

        r = np.sqrt((xcoord - xcenter)**2 + (ycoord - ycenter)**2)

        binary_circle = r <= numberpixel

        data = im[binary_circle]

        return binary_circle, data


    # Code beginning _____________________________________

    # ProcessImage Instance
    processing = cameracontrol.ProcessImage()

    # Path
    path_to_files = "/Volumes/KINGSTON/Quebec/Prototype/Exposure_gain_linear/linearity_20200303_2254"
    number_of_gain = 7
    files = []
    for n in range(number_of_gain):
        file = glob.glob(path_to_files + "/*_{0}.tif".format(str(n)))
        file.sort()
        files.append(file)

    # Finding center position
    im_c, met_c = processing.readTIFF_xiMU(files[0][5])
    im_c_dws = processing.dwnsampling(im_c, "BGGR")

    bin, regpro = processing.regionproperties(im_c_dws[:, :, 1], 0.75 * im_c_dws[:, :, 1].max())
    yc, xc = regpro[0].centroid
    print(xc, yc)

    # Pre-allocation
    alldata = {}

    for gainfile in files:

        dat = np.empty((len(gainfile), 7))

        for n, f in enumerate(gainfile):
            im, met = processing.readTIFF_xiMU(f)
            im -= int(met["black_level"])  # Taking dark each exposure time ?
            im_dws = processing.dwnsampling(im, "BGGR")

            dat[n, 0] = float(met["exposure_time_us"])

            # Printing info
            print("Exposure {0:.0f} us, Gain {1:.0f} dB, Board temperature {2:.3f} ˚C".format(met["exposure_time_us"],
                                                                                              met["gain_db"],
                                                                                              met["board temp"]))

            for i in range(im_dws.shape[2]):
                bin_circle, data = circle_mask(im_dws[:, :, i], xc, yc, 7)  # 10 pixels

                dat[n, i+1] = np.mean(data)
                dat[n, i+4] = np.std(data)

        alldata[str(int(met["gain_db"]))] = dat

    # Keys of alldata
    print(alldata.keys())
    print(alldata["0"])

    fig1, ax1 = plt.subplots(1, 3, figsize=(10, 10/2))

    for k in alldata.keys():
        for n in range(3):
            print(alldata[k][:, n+4])
            ax1[n].errorbar(alldata[k][:, 0], alldata[k][:, n+1], yerr=alldata[k][:, n+4], marker=".", label="{} dB".format(k))

    titl = ["610 nm", "530 nm", "470 nm"]
    for n in range(3):
        ax1[n].set_xlabel("Exposure time [micro seconds]")
        ax1[n].set_ylabel("Digital number")
        ax1[n].set_title(titl[n])
        ax1[n].legend(loc="best")

    plt.tight_layout()


    plt.show()
