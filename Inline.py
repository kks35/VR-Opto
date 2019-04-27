import cv2
import numpy as np
import os
import time

cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

flag = 1
num = 1

def make_1080p():
    cap_left.set(3, 1920.0)
    cap_left.set(4, 1080.0)
    cap_right.set(3, 1920.0)
    cap_right.set(4, 1908.0)

make_1080p()

def jie(x):
    val = 1
    for i in range(x):
        val *= x
        x -= 1
    return val


while(cap_left.isOpened() ):
    success_left, img_left = cap_left.read()
    success_right, img_right = cap_right.read()

    cv2.namedWindow("Cap_left", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Cap_right", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cap_left", 600, 600)
    cv2.resizeWindow("Cap_right", 600, 600)
    cv2.moveWindow("Cap_left", 0, 0)
    cv2.moveWindow("Cap_right",800, 0)

    cv2.imshow("Cap_left", img_left)
    cv2.imshow("Cap_right", img_right)
    
    k = cv2.waitKey(1)
    if k == ord("s"):
        cv2.imwrite("C://Users//MEC 101//Desktop//CAM//Inline//" + str(num) + ".jpg", img_left)
        cv2.imwrite("C://Users//MEC 101//Desktop//CAM//Inline//" + str(num) + "(2).jpg", img_right)
        print("save successful:" + str(num) + ".jpg     " + str(num) + "(2).jpg")
        print("_____________________")
        num += 1
        cv2.destroyAllWindows()
    elif k == 27:
        cv2.destroyAllWindows()
        break
cap_left.release()
cap_right.release()


    
    







