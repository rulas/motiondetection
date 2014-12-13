# coding=utf-8
from __future__ import division
"""
Created on Tue Sep 03 13:20:31 2013

This script process video capture and attempt to detect motion on that video

@author: rulas
"""
__author__ = 'rulas'


import time
import threading

import cv2
import cv
import numpy as np

from ringbuffer import RingBuffer
from alarm import IntrusionAlarm 

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
    # cv2.createTrackbar("DeltaThreshold", trackbarWindowName, 10, 32, void_func)
    # cv2.createTrackbar("HistoryLenght", trackbarWindowName, 10, 32, void_func)
    cv2.createTrackbar("Iterations", trackbarWindowName, 2, 16, void_func)
    cv2.createTrackbar("OpeningSize", trackbarWindowName, 3, 50, void_func)
    cv2.createTrackbar("ClosingSize", trackbarWindowName, 2, 50, void_func)
    cv2.createTrackbar("DilatingSize", trackbarWindowName, 2, 50, void_func)
    cv2.createTrackbar("AlarmThreshold", trackbarWindowName, 50, 100, void_func)


def get_trackbar_pos(trackbarname):
    return cv2.getTrackbarPos(trackbarname, trackbarWindowName)


def run_alarm():
    alarm = IntrusionAlarm()
    alarm.run()
    

#@profile
def main():
    """
    image processing occurs here
    """

    alarm_thread = None
    intrusion_detected = False
    alarm_hold_time = time.time() + 5

    frame_rate = 60
    refresh_time = int((1 / frame_rate) * 1000)

    ###########################################################################
    # VIDEO CAPTURE: open camera. user -1 for default. 1 for c920 or external 
    # camera.
    ###########################################################################
    vc = cv2.VideoCapture(-1)
    # vc = cv2.VideoCapture("Video 3.wmv")
    # vc = cv2.VideoCapture("test\motion2.mp4")

    ###########################################################################
    # create processing windows
    ###########################################################################
    cv2.namedWindow("src", flags=cv2.WINDOW_NORMAL)
    # cv2.namedWindow("gray", flags=cv2.WINDOW_NORMAL)
    # cv2.namedWindow("image_delta", flags=cv2.WINDOW_NORMAL)
    cv2.namedWindow("motion_mask", flags=cv2.WINDOW_NORMAL)
    cv2.namedWindow("BackgroundSubtractorMOG", flags=cv2.WINDOW_NORMAL)
    # cv2.namedWindow("noise removal", flags=cv2.WINDOW_NORMAL)

    ###########################################################################
    # create window with trackbars for runtime adjusting
    ###########################################################################
    create_trackbars()

    ###########################################################################
    # flush camera buffer. Don'alarm_thread quite understand why
    ###########################################################################
    for i in range(10):
        cv2.waitKey(refresh_time)
        _, src = vc.read()

    ###########################################################################
    # create a ring buffer to handle background history
    ###########################################################################
    prev_history_length = get_trackbar_pos("HistoryLenght")
    rb = RingBuffer(prev_history_length)

    #300, 2, 0.9, 1
    bgs_mog = cv2.BackgroundSubtractorMOG(
        history=500,
        nmixtures=6,
        backgroundRatio=.3,
        noiseSigma=1)
    # bgs_mog = cv2.BackgroundSubtractorMOG2(150, 128, True)

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
        # cv2.imshow("gray", gray)

        #######################################################################
        # remove noise from image
        #######################################################################
        #gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.blur(gray, (3, 3))
        #gray = cv2.medianBlur(gray, 5)
        #cv2.imshow("noise removal", gray)

        fgmask = bgs_mog.apply(gray)
        motion_mask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        cv2.imshow("BackgroundSubtractorMOG", motion_mask)

        #######################################################################
        # update background history, and background average
        #######################################################################
        # history_length = get_trackbar_pos("HistoryLenght")
        # history_length = 1 if history_length == 9 else history_length
        # if prev_history_length != history_length:
        #     rb = RingBuffer(history_length)
        #     prev_history_length = history_length
        # rb.push(gray.astype(np.float))

        # history = rb.buffer
        # gray_background = (sum(history) / len(history)).astype(np.uint8)
        # cv2.imshow("background", gray_background)

        #######################################################################
        # do background substraction
        #######################################################################
        # image_delta = abs(gray.astype(np.int8) - gray_background.astype(np.int8))
        # image_delta = image_delta.astype(np.uint8)
        # delta_threshold = get_trackbar_pos("DeltaThreshold")
        # _, image_delta = cv2.threshold(image_delta, delta_threshold, 255, cv2.THRESH_BINARY)
        # cv2.imshow("image_delta", image_delta)

        #######################################################################
        # erode and dilate image
        #######################################################################
        # get values from trackbars
        iterations = get_trackbar_pos("Iterations")
        open_size = get_trackbar_pos("OpeningSize")
        close_size = get_trackbar_pos("ClosingSize")        
        dilate_size = get_trackbar_pos("DilatingSize")        
        # validate data
        open_size = 1 if open_size == 0 else open_size
        close_size = 1 if close_size == 0 else close_size

        # opening image
        kernel = np.ones((open_size, open_size), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

        # closing image
        kernel = np.ones((close_size, close_size), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

        # dilate image
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations)
        cv2.imshow("motion_mask", motion_mask)

        #######################################################################
        # find and draw contours
        #######################################################################
        # contours, hierarchy = cv2.findContours(image_delta, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # print "num countours found = %s" % (len(contours))
        # cv2.drawContours(src, contours, -1, (0, 255, 0), -1)

        ###########################################################################
        # blob detection: detect blobs in image
        ###########################################################################
        params = cv2.SimpleBlobDetector_Params()
        # params.minDistBetweenBlobs = 10.0
        params.minArea = 500
        params.maxArea = 100000
        # params.filterByColor = 1
        # params.blobColor = 0
        params.minThreshold = 1
        params.maxThreshold = 255
        detector = cv2.SimpleBlobDetector(params)
        features = detector.detect(motion_mask)
        print "num features = %d" % len(features)
        for feature in detector.detect(motion_mask):
            x, y = feature.pt
            x, y = int(x), int(y)
            size = int(feature.size)

            cv2.rectangle(src, (x, y), (x+size, y+size), (255, 0, 0), 2)

        #######################################################################
        # Detect if Motion Alarm is to be triggered
        # hold off for 5 seconds until image stabilized
        #######################################################################

        # if alarm_hold_time < time.time(): 
        delta_percentage = 100 * np.count_nonzero(motion_mask) / motion_mask.size
        alarm_threshold = get_trackbar_pos("AlarmThreshold")

        if delta_percentage > alarm_threshold and not intrusion_detected:
            alarm_thread = threading.Thread(target=run_alarm)
            alarm_thread.start()
            intrusion_detected = True   

        #######################################################################
        # image info
        #######################################################################
        # import pdb; pdb.set_trace()
        # if alarm_hold_time < time.time():
        pos = (5, 15)
        info = "motion_rate = %0.2f, threshold = %0.2f" % (delta_percentage, alarm_threshold)
        cv2.putText(src, info, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness=1, lineType=cv2.CV_AA)

        #######################################################################
        # display video processing output
        #######################################################################
        cv2.SimpleBlobDetector()
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
        if key == ord('a'):
            import pdb; pdb.set_trace()

    if intrusion_detected:
        alarm_thread.join()

if __name__ == "__main__":
    main()
