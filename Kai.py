import cv2
import numpy as np
import itertools
import math
import cv2.cv as cv
import glob
import os

#draw函数用于在棋盘上划线
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)    
img = glob.glob("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\Veri\\Dst2\\*.jpg")

objp = np.zeros((6*6,3),np.float32)
objp[:,:2] = np.mgrid[0:6,0:6].T.reshape(-1,2)

obj_points = [] #3D points
img_points = [] #2D points

for fname in img:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2. findChessboardCorners(gray, (6,6), None)

    if ret == True:
        cv2.cornerSubPix(gray, corners, (11,11),(-1,-1),criteria)
        obj_points.append(objp)
        img_points.append(corners)

#是否需要标出在图像上
"""
        cv2.drawChessboardCorners(img, (6,8),corners,ret)
        cv2.namedWindow("FindCorners", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("FindCorners", 600,600)
        cv2.imshow("FindCorners", img)
        cv2.waitKey(0)
cv2.destroyAllWindows()
"""


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
fx = mtx[0][0]
fy = mtx[1][1]
cx = mtx[0][2]
cy = mtx[1][2]
k1, k2, k3, p1, p2 = dist[0]
mrot,_ = cv2.Rodrigues(rvecs[0])


print " fx, fy ",fx, fy
print " cx, cy ",cx, cy
print " k1, k2, k3, p1, p2 ",k1,k2,k3,p1,p2
print " Tvecs ",tvecs
print " Rvecs ",rvecs
print " mrot ",mrot

#Undist
IMG = glob.glob("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\118\\Checkboard\\*.jpg")
img_save = "C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\Veri\\Dst"
for fname in IMG:

    Img = cv2.imread(fname)
    IMG = cv.CreateImage((Img.shape[1],Img.shape[0]),8,3)
    (img_dir, tempfilename) = os.path.split(fname) # 分离文件目录 文件名和文件后缀
    
    h,w = Img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist,(6,6),1,(6,6))
    
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (6,6),5)
    dst = cv2.remap(Img, mapx,  mapy, cv2.INTER_LINEAR)
"""    
    x,y,w,h = roi
    #dst = dst[y:y+h, x:x+w]
"""
    
#是否写出去畸变后的文件
"""    
    savepath = os.path.join(img_save, tempfilename)
    cv2.imwrite(savepath, dst)
"""
    




#project points and find reprojection error
sqsum2_error = 0
mean_error = 0
for i in xrange(len(obj_points)):
    imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error_sqsum = cv2.norm(img_points[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    error_mean = cv2.norm(img_points[i],imgpoints2, cv2.NORM_L1)/len(imgpoints2)
    sqsum2_error += error_sqsum
    mean_error += error_mean
    

print "sqrt sum square error: ", sqsum2_error/len(obj_points)
print "mean error: ", mean_error/len(obj_points)


