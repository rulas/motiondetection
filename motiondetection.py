# coding=utf-8
"""
Created on Tue Sep 03 13:20:31 2013

This script process video capture and attempt to detect motion on that video

@author: rulas
"""
__author__ = 'rulas'

from __future__ import division
import time

import cv2
import cv
import numpy as np

from ringbuffer import RingBuffer

trackbarWindowName = "filters"


def void_func(number):
    """
    do nothing
    :param number:
    """
    pass


def create_trackbars():
    """
    creates trackbars for fine image processing tuning

    """
    # create window for trackbars
    cv2.namedWindow(trackbarWindowName, flags=cv.CV_WINDOW_NORMAL)
    cv2.createTrackbar("DeltaThreshold", trackbarWindowName, 10, 32, void_func)
    cv2.createTrackbar("HistoryLenght", trackbarWindowName, 10, 32, void_func)
    cv2.createTrackbar("Iterations", trackbarWindowName, 2, 16, void_func)
    cv2.createTrackbar("MorphOpsSize", trackbarWindowName, 2, 16, void_func)


def get_trackbar_pos(trackbarname):
    return cv2.getTrackbarPos(trackbarname, trackbarWindowName)

#@profile
def main():
    """
    image processing occurs here
    """
    frame_rate = 60
    refresh_time = int((1 / frame_rate) * 1000)
    ###########################################################################
    # open camera. user -1 for default. 1 for c920 or external camera.
    ###########################################################################
    #vc = cv2.VideoCapture(1)
    vc = cv2.VideoCapture("Video 3.wmv")

    ###########################################################################
    # create processing windows
    ###########################################################################
    cv2.namedWindow("src")
    cv2.namedWindow("gray")
    cv2.namedWindow("image_delta")
    cv2.namedWindow("image_delta_open")
    cv2.namedWindow("noise removal")

    ###########################################################################
    # create window with trackbars for runtime adjusting
    ###########################################################################
    create_trackbars()

    ###########################################################################
    # flush camera buffer. Don't quite understand why
    ###########################################################################
    for i in range(10):
        cv2.waitKey(refresh_time)
        _, src = vc.read()

    ###########################################################################
    # create a ring buffer to handle background history
    ###########################################################################
    prev_history_length = get_trackbar_pos("HistoryLenght")
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
        # update background history, and background average
        #######################################################################
        history_length = get_trackbar_pos("HistoryLenght")
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
        delta_threshold = get_trackbar_pos("DeltaThreshold")
        _, image_delta = cv2.threshold(image_delta, delta_threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow("image_delta", image_delta)


        #######################################################################
        # open image
        #######################################################################
        size = get_trackbar_pos("MorphOpsSize")
        size = 1 if size == 0 else size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        iterations = get_trackbar_pos("Iterations")
        iterations = 1 if size == 0 else iterations
        image_delta = cv2.morphologyEx(image_delta, cv2.MORPH_OPEN, kernel, iterations=iterations)

        #######################################################################
        # dilate image
        #######################################################################
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        image_delta = cv2.dilate(image_delta, element)
        cv2.imshow("image_delta_open", image_delta)

        #######################################################################
        # find and draw contours
        #######################################################################
        contours, hierarchy = cv2.findContours(image_delta, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print "num countours found = %s" % (len(contours))
        cv2.drawContours(src, contours, -1, (0, 255, 0), -1)

        ###########################################################################
        # blob detection: detect blobs in image
        ###########################################################################
        pass

        #######################################################################
        # display video processing output
        #######################################################################
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