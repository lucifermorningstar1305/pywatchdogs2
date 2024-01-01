"""
Citation: https://github.com/Sentdex/pygta5/blob/master/original_project/1.%20collect_data.py
"""


import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os

# w = [1,0,0,0,0,0,0,0,0]
# s = [0,1,0,0,0,0,0,0,0]
# a = [0,0,1,0,0,0,0,0,0]
# d = [0,0,0,1,0,0,0,0,0]
# wa = [0,0,0,0,1,0,0,0,0]
# wd = [0,0,0,0,0,1,0,0,0]
# sa = [0,0,0,0,0,0,1,0,0]
# sd = [0,0,0,0,0,0,0,1,0]
# nk = [0,0,0,0,0,0,0,0,1]

w, s, a, d, wa, wd, sa, sd, nk = 0, 1, 2, 3, 4, 5, 6, 7, 8

starting_value = 1

while True:
    file_name = "./data/phase_1/training_data-{}.npy".format(starting_value)

    if os.path.isfile(file_name):
        print("File exists, moving along", starting_value)
        starting_value += 1
    else:
        print("File does not exist, starting fresh!", starting_value)

        break


def keys_to_output(keys):
    """
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    """
    # output = [0,0,0,0,0,0,0,0,0]
    output = None

    if "W" in keys and "A" in keys:
        output = wa
    elif "W" in keys and "D" in keys:
        output = wd
    elif "S" in keys and "A" in keys:
        output = sa
    elif "S" in keys and "D" in keys:
        output = sd
    elif "W" in keys:
        output = w
    elif "S" in keys:
        output = s
    elif "A" in keys:
        output = a
    elif "D" in keys:
        output = d
    else:
        output = nk
    return output


def main(file_name, starting_value):
    if not os.path.exists("./data/"):
        os.mkdir("./data/")

    file_name = file_name
    starting_value = starting_value
    training_data = []
    for i in list(range(10))[::-1]:
        print(i + 1)
        time.sleep(1)

    last_time = time.time()
    paused = False
    print("STARTING!!!")

    counter = 1
    while True:
        if not paused:
            screen = grab_screen(region=(0, 40, 1024, 768))
            last_time = time.time()
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (512, 512))
            # run a color convert:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            keys = key_check()
            output = keys_to_output(keys)

            print(output)

            cv2.imwrite(f"./data/img{counter}_{output}.png", screen)
            counter += 1

            if counter % 100 == 0:
                print(f"Number of images collected: {counter}")

        keys = key_check()
        if "T" in keys:
            if paused:
                paused = False
                print("unpaused!")
                time.sleep(1)
            else:
                print("Pausing!")
                paused = True
                time.sleep(1)


main(file_name, starting_value)
