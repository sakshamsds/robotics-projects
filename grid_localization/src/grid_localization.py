#!/usr/lib/bin/env/ python

import rospy
import numpy as np
from std_msgs.msg import String
import time

import matplotlib.pyplot as plt

plt.ion()

MAP = np.zeros(shape=(20, 20))
MAP[10, 10] = 1

MOTION_COMMANDS = ["R", "L", "U", "D"]
OBSERVATION_COMMANDS = ["X", "Y"]


plt.figure(figsize=(12, 12))
plt.xticks(np.arange(0, 20, 1))
plt.yticks(np.arange(0, 20, 1))
display_image = plt.imshow(MAP)
plt.gca().invert_yaxis()
display_image.set_data(MAP)
plt.colorbar()
# plt.pause(0.00000000000000000001)
plt.pause(0.0001)


def observation(axis, coordinate):
    if axis == "X":
        MAP[:, coordinate] *= 0.4
        if coordinate == 19:
            MAP[:, :17] = 0
            MAP[:, 17] *= 0.1
            MAP[:, 18] *= 0.2
        elif coordinate == 18:
            MAP[:, :16] = 0
            MAP[:, 16] *= 0.1
            MAP[:, 17] *= 0.2
            MAP[:, 19] *= 0.2
        elif coordinate == 17:
            MAP[:, :15] = 0
            MAP[:, 15] *= 0.1
            MAP[:, 19] *= 0.1
            MAP[:, 16] *= 0.2
            MAP[:, 18] *= 0.2
        elif coordinate == 0:
            MAP[:, 3:] = 0
            MAP[:, 2] *= 0.1
            MAP[:, 1] *= 0.2
        elif coordinate == 1:
            MAP[:, 4:] = 0
            MAP[:, 3] *= 0.1
            MAP[:, 2] *= 0.2
            MAP[:, 0] *= 0.2
        elif coordinate == 2:
            MAP[:, 5:] = 0
            MAP[:, 4] *= 0.1
            MAP[:, 0] *= 0.1
            MAP[:, 3] *= 0.2
            MAP[:, 1] *= 0.2
        else:
            MAP[:, coordinate - 1] *= 0.2
            MAP[:, coordinate + 1] *= 0.2
            MAP[:, coordinate - 2] *= 0.1
            MAP[:, coordinate + 2] *= 0.1
            MAP[:, coordinate + 3 :] = 0
            MAP[:, : coordinate - 2] = 0
        normalize_X()

    elif axis == "Y":
        MAP[coordinate, :] *= 0.4
        if coordinate == 19:
            MAP[:17, :] = 0
            MAP[17, :] *= 0.1
            MAP[18, :] *= 0.2
        elif coordinate == 18:
            MAP[:16, :] = 0
            MAP[16, :] *= 0.1
            MAP[17, :] *= 0.2
            MAP[19, :] *= 0.2
        elif coordinate == 17:
            MAP[:15, :] = 0
            MAP[15, :] *= 0.1
            MAP[19, :] *= 0.1
            MAP[16, :] *= 0.2
            MAP[18, :] *= 0.2
        elif coordinate == 0:
            MAP[3:, :] = 0
            MAP[2, :] *= 0.1
            MAP[1, :] *= 0.2
        elif coordinate == 1:
            MAP[4:, :] = 0
            MAP[3, :] *= 0.1
            MAP[2, :] *= 0.2
            MAP[0, :] *= 0.2
        elif coordinate == 2:
            MAP[5:, :] = 0
            MAP[4, :] *= 0.1
            MAP[0, :] *= 0.1
            MAP[3, :] *= 0.2
            MAP[1, :] *= 0.2
        else:
            MAP[coordinate - 1, :] *= 0.2
            MAP[coordinate + 1, :] *= 0.2
            MAP[coordinate - 2, :] *= 0.1
            MAP[coordinate + 2, :] *= 0.1
            MAP[coordinate + 3 :, :] = 0
            MAP[: coordinate - 2, :] = 0
        normalize_Y()

    return


def motion(direction, steps):
    if direction == "R":
        for _ in range(steps):
            for i in range(19, 0, -1):
                if i == 0:
                    MAP[:, 0] += 0.2 * MAP[:, 0]
                elif i == 1:
                    MAP[:, 1] += 0.6 * MAP[:, 0] + 0.2 * MAP[:, 1]
                else:
                    MAP[:, i] += (
                        0.2 * MAP[:, i] + 0.6 * MAP[:, i - 1] + 0.2 * MAP[:, i - 2]
                    )
        normalize_X()

    elif direction == "L":
        for _ in range(steps):
            for i in range(0, 19, 1):
                if i == 19:
                    MAP[:, 19] += 0.2 * MAP[:, 19]
                elif i == 18:
                    MAP[:, 18] += 0.6 * MAP[:, 19] + 0.2 * MAP[:, 18]
                else:
                    MAP[:, i] += (
                        0.2 * MAP[:, i] + 0.6 * MAP[:, i + 1] + 0.2 * MAP[:, i + 2]
                    )
        normalize_X()

    elif direction == "U":
        for _ in range(steps):
            for i in range(19, 0, -1):
                if i == 0:
                    MAP[0, :] += 0.2 * MAP[0, :]
                elif i == 1:
                    MAP[1, :] += 0.6 * MAP[0, :] + 0.2 * MAP[1, :]
                else:
                    MAP[i, :] += (
                        0.2 * MAP[i, :] + 0.6 * MAP[i - 1, :] + 0.2 * MAP[i - 2, :]
                    )
        normalize_Y()

    elif direction == "D":
        for _ in range(steps):
            for i in range(0, 19, 1):
                if i == 19:
                    MAP[19, :] += 0.2 * MAP[19, :]
                elif i == 18:
                    MAP[18, :] += 0.6 * MAP[19, :] + 0.2 * MAP[18, :]
                else:
                    MAP[i, :] += (
                        0.2 * MAP[i, :] + 0.6 * MAP[i + 1, :] + 0.2 * MAP[i + 2, :]
                    )
        normalize_Y()

    return


def normalize_X():
    for row in range(20):
        total = np.sum(MAP[row, :])
        if total != 0:
            MAP[row, :] /= total
    return


def normalize_Y():
    for column in range(20):
        total = np.sum(MAP[:, column])
        if total != 0:
            MAP[:, column] /= total
    return


def callback(msg):
    # start = time.time()
    msg = msg.upper()
    print(msg)

    if msg[0] in MOTION_COMMANDS:
        direction = msg[0]
        steps = int(msg[1:])
        motion(direction, steps)
    elif msg[0] in OBSERVATION_COMMANDS:
        axis = msg[0]
        coordinate = int(msg[1:])
        observation(axis, coordinate)

    # display the updated map after each iteration
    display_image.set_data(MAP)
    plt.gcf().canvas.flush_events()
    # print("execution time:", round(time.time() - start, 3))
    return


def driver():
    rospy.init_node("grid_localization", anonymous=True)
    rospy.Subscriber("robot", data_class=String, callback=callback)

    # TEST, R4 -> X13 -> L8 -> X6 -> U4 -> Y13 -> D12 -> Y2
    # time.sleep(2)
    # callback("R4")
    # time.sleep(2)
    # callback("X13")
    # time.sleep(2)
    # callback("L8")
    # time.sleep(2)
    # callback("X6")
    # time.sleep(2)
    # callback("U4")
    # time.sleep(2)
    # callback("Y13")
    # time.sleep(2)
    # callback("D12")
    # time.sleep(2)
    # callback("Y2")
    # time.sleep(2)

    rospy.spin()
    return


# def gaussian_distribution(x, mu, sigma):
#     return np.exp(-0.5 * np.square((x - mu) / sigma)) / (np.sqrt(2 * np.pi) * sigma)


# # print(gaussian_distribution(-1, 0, 1))


if __name__ == "__main__":
    try:
        driver()
    except rospy.ROSInterruptException:
        pass
