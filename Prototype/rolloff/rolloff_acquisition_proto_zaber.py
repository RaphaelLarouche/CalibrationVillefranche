# -*- coding: utf-8 -*-

if __name__ == "__main__":

    import datetime
    import numpy as np
    import time

    from zaber_motion.binary import Connection
    from zaber_motion import Units

    import cameracontrol.cameracontrol as cc

    def saveimg(pathfolder, raw_image, metadata):
        # Saving image
        today = datetime.datetime.utcnow()
        imname = "/IMG_" + today.strftime("%Y%m%d_%H%M%S_UTC") + ".tif"
        path = pathfolder + imname

        cc.ProcessImage.saveTIFF_xiMU(path, raw_image, metadata)


    # zero_position = 135
    # pos_init = -105
    # pos_max = 105
    # degree_increment = 5
    #
    # num = int((pos_max - pos_init)/degree_increment) + 1
    #
    # pathfolder = cc.ProcessImage.folder_choice()
    #
    # with Connection.open_serial_port("/dev/cu.USA19H1411P1.1") as connection:
    #     device_list = connection.detect_devices()
    #     print("Found {} devices".format(len(device_list)))
    #
    #     # Initialization
    #     device = device_list[0]
    #     device.home()
    #     device.move_absolute(zero_position+pos_init, Units.ANGLE_DEGREES)
    #
    #     camera_object = cc.TakeImage(imageformat="raw")
    #
    #     for n in range(1):
    #         print("Image number {0} out of {1} ".format(n, num))
    #
    #         time.sleep(2)
    #
    #         raw_image, metadata = camera_object.acquisition(exposuretime=50000, gain=0, binning="2x2", video=False)
    #
    #         saveimg(pathfolder, raw_image, metadata)
    #
    #         time.sleep(1)
    #         device.move_relative(degree_increment, Units.ANGLE_DEGREES)
    #
    #     camera_object.end()

    #Control of position for alignment
    pos_wanted = 0
    zero_position = 135
    with Connection.open_serial_port("/dev/cu.USA19H1411P1.1") as connection:
        device_list = connection.detect_devices()
        print("Found {} devices".format(len(device_list)))

        # Initialization
        device = device_list[0]
        #device.home()
        device.move_absolute(zero_position + pos_wanted, Units.ANGLE_DEGREES)
