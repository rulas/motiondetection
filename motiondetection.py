# -*- coding: utf-8 -*-
"""
Created on Tue Sep 03 13:20:31 2013

This script process video capture and attempt to recognize a glue stick
with yellow tap. what an useful thing :D

@author: lrvillan
"""

from __future__ import division
import time

import cv2
import cv
from docutils.nodes import image
import numpy as np

from ringbuffer import RingBuffer

# initial min and max HSV filter values
# these will be changed using trackbars
H_MIN = 0
H_MAX = 255
S_MIN = 0
S_MAX = 255
V_MIN = 0
V_MAX = 255
trackbarWindowName = "filters"


def void_func(number):
    pass


def createTrackbars():
    """
    creates trackbars for fine image processing tuning

    """
    # create window for trackbars
    cv2.namedWindow(trackbarWindowName, flags=cv.CV_WINDOW_NORMAL)

    # create trackbars and insert them into window
    # 3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
    # the max value the trackbar can move (eg. H_HIGH),
    # and the function that is called whenever the trackbar is moved(eg. on_trackbar)

    #    cv2.createTrackbar("H_MIN", trackbarWindowName, H_MIN, H_MAX, void_func)
    #    cv2.createTrackbar("H_MAX", trackbarWindowName, H_MAX, H_MAX, void_func)
    #    cv2.createTrackbar("S_MIN", trackbarWindowName, S_MIN, S_MAX, void_func)
    #    cv2.createTrackbar("S_MAX", trackbarWindowName, S_MAX, S_MAX, void_func)
    #    cv2.createTrackbar("V_MIN", trackbarWindowName, V_MIN, V_MAX, void_func)
    #    cv2.createTrackbar("V_MAX", trackbarWindowName, V_MAX, V_MAX, void_func)

    cv2.createTrackbar("ErodeKernSize", trackbarWindowName, 5, 100, void_func)
    cv2.createTrackbar("DilateKernSize", trackbarWindowName, 5, 100, void_func)

    #    cv2.createTrackbar("KernelMultiplier", trackbarWindowName, 0, 100, void_func)

    cv2.createTrackbar("DeltaThreshold", trackbarWindowName, 10  , 32, void_func)
    cv2.createTrackbar("HistoryLenght", trackbarWindowName, 10, 32, void_func)
    cv2.createTrackbar("Iterations", trackbarWindowName, 2, 16, void_func)
    cv2.createTrackbar("MorphOpsSize", trackbarWindowName, 2, 16, void_func)
    cv2.createTrackbar("CannyLowThresh", trackbarWindowName, 1, 16, void_func)



def getTrackbars():
    h_min = cv2.getTrackbarPos("H_MIN", trackbarWindowName)
    h_max = cv2.getTrackbarPos("H_MAX", trackbarWindowName)
    s_min = cv2.getTrackbarPos("S_MIN", trackbarWindowName)
    s_max = cv2.getTrackbarPos("S_MAX", trackbarWindowName)
    v_min = cv2.getTrackbarPos("V_MIN", trackbarWindowName)
    v_max = cv2.getTrackbarPos("V_MAX", trackbarWindowName)

    return (h_min, h_max, s_min, s_max, v_min, v_max)


def getTrackbarPos(trackBarName):
    return cv2.getTrackbarPos(trackBarName, trackbarWindowName)


def getErodeSize():
    return cv2.getTrackbarPos("ErodeKernSize", trackbarWindowName)


def getDilateSize():
    return cv2.getTrackbarPos("DilateKernSize", trackbarWindowName)


def getKernelMultiplier():
    return cv2.getTrackbarPos("KernelMultiplier", trackbarWindowName)


def getDeltaThreshold():
    return cv2.getTrackbarPos("DeltaThreshold", trackbarWindowName)


def erode(src, iterations=1):
    size = getErodeSize() + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dst = cv2.erode(src, kernel, iterations=iterations)
    return dst


def dilate(src, iterations=1):
    size = getDilateSize() + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dst = cv2.dilate(src, kernel, iterations=iterations)
    return dst

#@profile
def main():
    frame_rate = 60
    refresh_time = int((1 / frame_rate) * 1000)
    ###########################################################################
    # open camera. user -1 for default. 1 for c920 or external camera.
    ###########################################################################
    #    vc = cv2.VideoCapture(1)
    vc = cv2.VideoCapture("Video 3.wmv")
    #    vc.set(3, 1280)
    #    vc.set(4, 720)

    ###########################################################################
    # create processing windows
    ###########################################################################
    cv2.namedWindow("src")
    cv2.namedWindow("gray")
    cv2.namedWindow("image_delta")
    cv2.namedWindow("image_delta_open")
    cv2.namedWindow("noise removal")
    cv2.namedWindow("background")
    #cv2.namedWindow("contours")
    #cv2.namedWindow("gray")
    #cv2.namedWindow("delta")

    ###########################################################################
    # create window with trackbars for runtime adjusting
    ###########################################################################
    createTrackbars()

    ###########################################################################
    # wait until a key is pressed.
    ###########################################################################
    #    while cv2.waitKey(1000) == -1:
    #        pass

    ###########################################################################
    # flush camera buffer. Don't quite understand why
    ###########################################################################
    for i in range(10):
        cv2.waitKey(refresh_time)
        _, src = vc.read()
    #    time.sleep(5)

    ###########################################################################
    # calculate the running average from the camera for n times
    ###########################################################################
    #    background = None
    #    HISTORY = 40
    #    background = np.zeros((480, 640), np.float)
    #
    #    for i in range(HISTORY):
    #        cv2.waitKey(40)
    #        _, src = vc.read()
    #        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    #        src = src.astype(np.float)
    #        background = (background + src.astype(np.float))/2
    #
    #    gray_background = background.astype(np.uint8)

    ###########################################################################
    # create a ring buffer to handle background history
    ###########################################################################
    prev_history_length = getTrackbarPos("HistoryLenght")
    rb = RingBuffer(prev_history_length)

    ###########################################################################
    # process image
    ###########################################################################
    print "start processing image"
    while True:
        #######################################################################
        # start measuring time
        #######################################################################
        start_time = time.time()

        #######################################################################
        # read data directly from the camera or video
        #######################################################################
        _, src = vc.read()

        #######################################################################
        # convert to hsv and gray frames
        #######################################################################
        #        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray", gray)


        #######################################################################
        # remove noise from image
        #######################################################################
        #gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.blur(gray, (3, 3))
        #gray = cv2.medianBlur(gray, 5)
        cv2.imshow("noise removal", gray)

        #######################################################################
        # read values from HSV trackbars
        #######################################################################
        # max values
        #        h_min, h_max, s_min, s_max, v_min, v_max = getTrackbars()
        #        hsv_min = np.array([h_min, s_min, v_min], np.uint8)
        #        hsv_max = np.array([h_max, s_max, v_max], np.uint8)

        #######################################################################
        # create a new image that results from thresholding the image with
        # previous hsv limits. then erode and dilate using trackbar values
        #######################################################################
        #        threshold = cv2.inRange(hsv, hsv_min, hsv_max)
        #        threshold = erode(threshold)
        #        threshold = dilate(threshold)

        #######################################################################
        # update background history, and background average
        #######################################################################
        history_length = getTrackbarPos("HistoryLenght")
        history_length = 1 if history_length == 9 else history_length
        if prev_history_length != history_length:
            rb = RingBuffer(history_length)
            prev_history_length = history_length
        rb.push(gray.astype(np.float))

        history = rb.buffer
        gray_background = (sum(history) / len(history)).astype(np.uint8)
        cv2.imshow("background", gray_background)

        #######################################################################
        # do background substraction
        #######################################################################
        image_delta = abs(gray.astype(np.int8) - gray_background.astype(np.int8))
        image_delta = image_delta.astype(np.uint8)
        deltaThreshold = getTrackbarPos("DeltaThreshold")
        _, image_delta = cv2.threshold(image_delta, deltaThreshold, 255, cv2.THRESH_BINARY)
        cv2.imshow("image_delta", image_delta)


        #######################################################################
        # open image
        #######################################################################
        #        image_delta = erode(image_delta, 2)
        #        image_delta = dilate(image_delta, 2
        size = getTrackbarPos("MorphOpsSize")
        size = 1 if size == 0 else size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        iterations = getTrackbarPos("Iterations")
        iterations = 1 if size == 0 else iterations
        image_delta = cv2.morphologyEx(image_delta, cv2.MORPH_OPEN, kernel, iterations=iterations)

        #######################################################################
        # dilate image
        #######################################################################
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        image_delta = cv2.dilate(image_delta, element)
        cv2.imshow("image_delta_open", image_delta)

        #######################################################################
        # find contours
        #######################################################################
        #motion = cv2.bitwise_not(image_delta)

        #contours, hierarchy = cv2.findContours(image_delta, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(image_delta, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print "num countours found = %s" % (len(contours))
        #for contour in contours:
        #cv2.drawContours(image=des, contours=[contour], contourldx=-1, externalColor=(0, 255, 0), thickness=-1)
        cv2.drawContours(src, contours, -1, (0, 255, 0), -1)

        #motion = cv2.bitwise_not(motion)

        ###########################################################################
        # blob detection: detect blobs in image
        ###########################################################################
        pass

        ###########################################################################
        # draw a colored square indicating moving objects in original image
        ###########################################################################
        blobs = []
        for each_blob in blobs:
            # draw blob in origal image
            pass



        #######################################################################
        # display video processing output
        #######################################################################
        #        cv2.imshow("src", src)
        #        cv2.imshow("hsv", hsv)
        #cv2.imshow("gray", gray)
        ##        cv2.imshow("threshold", threshold)
        #cv2.imshow("delta", image_delta)
        cv2.imshow("src", src)

        #######################################################################
        # calculate elapsed time
        #######################################################################
        elapsed_time = time.time() - start_time
        print "processing time = %.2sms, frame rate = %dfps" % (elapsed_time * 1000, refresh_time)

        #######################################################################
        # cv2.waitkey expects a value that corresponds to the msecs to wait
        # which finally defines the frame rate of video.
        #######################################################################
        key = cv2.waitKey(refresh_time)

        #######################################################################
        # stop video processing if user press ESC key
        #######################################################################
        if key == 27:
            vc.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()