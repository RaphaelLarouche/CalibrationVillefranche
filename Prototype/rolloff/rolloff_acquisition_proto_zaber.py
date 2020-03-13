# -*- coding: utf-8 -*-

import datetime
import time

from zaber_motion.binary import Connection
from zaber_motion import Units

import cameracontrol.cameracontrol as cc


def saveimg(pathfolder, raw_image, metadata, dark=False):
    # Saving image
    today = datetime.datetime.utcnow()
    imname = "/IMG_" + today.strftime("%Y%m%d_%H%M%S_UTC") + ".tif"
    if dark:
        imname = "/DARK_" + today.strftime("%Y%m%d_%H%M%S_UTC") + ".tif"
    path = pathfolder + imname

    cc.ProcessImage.saveTIFF_xiMU(path, raw_image, metadata)


def timer(t):
    while t > 0:
        print("{0} seconds before dark".format(t))
        t -= 1
        time.sleep(1)


if __name__ == "__main__":

    zero_position = 135
    pos_init = -105
    pos_max = 105
    degree_increment = 5

    num = int((pos_max - pos_init)/degree_increment) + 1

    pathfolder = cc.ProcessImage.folder_choice()

    with Connection.open_serial_port("/dev/cu.USA19H1411P1.1") as connection:
        device_list = connection.detect_devices()
        print("Found {} devices".format(len(device_list)))

        # Initialization
        device = device_list[0]
        device.home()
        device.move_absolute(zero_position+pos_init, Units.ANGLE_DEGREES)

        camera_object = cc.TakeImage(imageformat="raw")

        for n in range(num):
            print("Image number {0} out of {1} ".format(n+1, num))

            time.sleep(2)

            raw_image, metadata = camera_object.acquisition(exposuretime=72600, gain=0, binning="2x2", video=False)

            saveimg(pathfolder, raw_image, metadata)

            time.sleep(1)
            device.move_relative(degree_increment, Units.ANGLE_DEGREES)

        # Dark image 30 seconds after
        timer(30)
        print("Taking dark image.")
        dark_image, dark_metadata = camera_object.acquisition(exposuretime=72600, gain=0, binning="2x2", video=False)
        saveimg(pathfolder, dark_image, dark_metadata, dark=True)

        camera_object.end()
