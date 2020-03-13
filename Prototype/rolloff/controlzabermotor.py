# -*- coding: utf-8 -*-
"""
Control of Zaber motor for alignment mostly.
"""

from zaber_motion.binary import Connection
from zaber_motion import Units


def zaber_motor(port, angle, home=True):
    """
    Zaber motor control.

    :param port:
    :param angle:
    :param home:
    :return:
    """

    zero_position = 135

    with Connection.open_serial_port(port) as connection:
        device_list = connection.detect_devices()
        print("Found {} devices".format(len(device_list)))

        # Initialization
        device = device_list[0]
        if home:
            device.home()
        device.move_absolute(zero_position + angle, Units.ANGLE_DEGREES)


if __name__ == "__main__":

    zaber_motor("/dev/cu.USA19H1411P1.1", 0, home=False)
