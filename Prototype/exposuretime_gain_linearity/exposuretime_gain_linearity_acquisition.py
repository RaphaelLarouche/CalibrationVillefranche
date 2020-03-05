"""
Script for the data acquisition of experiment regarding the exposure time and gain linearity.
"""

if __name__ == "__main__":

    # Importation of standard modules
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import time
    import datetime

    # Importation of other modules
    import cameracontrol.cameracontrol as cameracontrol
    from ximea import xiapi
    import tifffile.tifffile

    # Functions
    def acquisition(cam, img, exposure, gain):

        cam.set_gain(gain)
        cam.set_exposure(exposure)

        cam.get_image(img)

        data_raw = img.get_image_data_numpy()
        data_raw = data_raw[::-1, :]

        met = cameracontrol.TakeImage.metadata_xiMU(img)

        # Adding board temp
        temp = cam.get_sensor_board_temp()
        met["board temp"] = temp

        return data_raw, met

    def save(pathfolder, image, metadata, gainnumber, dark=False):

        today = datetime.datetime.utcnow()

        if dark:
            imname = "/DARK_" + today.strftime("%Y%m%d_%H%M%S_UTC_") + str(gainnumber) + ".tif"  # year/month/day
        else:
            imname = "/IMG_" + today.strftime("%Y%m%d_%H%M%S_UTC_") + str(gainnumber) + ".tif"  # year/month/day

        path = pathfolder + imname

        if len(image.shape) == 2:
            tifffile.imwrite(path, image.astype(int), metadata=metadata)
        else:
            raise ValueError("Only raw bayer mosaic image can be saved.")

    # Code beginning _____________________________________

    # Ximea camera and image instance
    cam = xiapi.Camera()
    img = xiapi.Image()

    # Opening camera
    cam.open_device_by("XI_OPEN_BY_SN", "16990159")

    # Setting image parameters
    cam.set_imgdataformat("XI_RAW16")  # Image format
    cam.set_downsampling_type("XI_BINNING")
    cam.set_downsampling("XI_DWN_2x2")

    # Creating a directory
    now = datetime.datetime.utcnow()
    cwd = os.getcwd()
    path = cwd + "/linearity_" + now.strftime("%Y%m%d_%H%M")
    os.makedirs(path)

    # Exposure (between 72 mus and 10 000 000 mus)
    max_exposure = [67388, 39969, 22001, 15150, 7183, 3670, 2345]  # microseconds  # To be changed 
    exposure_multi = np.linspace(0, 1, 11)  # 10 points
    exposure_multi = exposure_multi[1::]

    # Gain (between -4 and 38 dB)
    gain = [0, 5, 10, 15, 20, 25, 30]

    # Starting acquisition
    print("Starting data acquisition...")
    cam.start_acquisition()

    for n, current_gain in enumerate(gain):
        # DARK ?
        for current_exp in exposure_multi * max_exposure[n]:

            current_gain = int(current_gain)
            current_exp = int(current_exp)
            print("Current gain: {0:.3f}, Current exposure: {1:.3f}".format(current_gain, current_exp))
            im, met = acquisition(cam, img, current_exp, current_gain)
            save(path, im, met, n)

            time.sleep(2)  # 2 seconds

    # Stopping acquisition
    cam.stop_acquisition()
    print("Acquisition stopped...")

    cam.close_device()

    # Show last image
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    ax1.imshow(im)
    plt.show()
