import cv2
import numpy as np


camera_L_mtx = np.array([[1475.30861390889, 923.124697556223, 565.300172727283],
                               [0.0, 1473.96139905513, 0.0],
                               [0.0, 0.0, 1.0]])
dist_L = np.array([[-0.00271560252839921, 0.0109174756408686, 0.0, 0.0, -0.0509649204269123]])



camera_R_mtx = np.array([[1019.97588832716, 632.218094556863, 374.363508177541],
                                [0.0, 1019.05710298783, 0.0],
                                [0.0, 0.0, 1.0]])
dist_R = np.array([[0.112677018102869, -0.365453759593591, 0.0, 0.0, 0.472952092490304]])

R = np.array([0.995700037775415, 0.0255943758662884, -0.0890301224196135],
             [-0.0262634196054573, 0.999634880231602, -0.00635130025881020],
             [0.0888350581958716, 0.00866222537025188, 0.996008683841146])
T = np.array([-81.6740559259621, -39.7592429266074, 2.91674481347275]) # 平移关系向量

size_L = (1920, 1080) # 图像尺寸
size_R = (1280, 720)

R1_L, R2_L, P1_L, P2_L, Q_L = cv2.stereoRectify(camera_L_mtx, dist_L, camera_R_mtx, dist_R, size_L, R, T) # for cam L R1 矫正变换旋转矩阵
R1_R, R2_R, P1_R, P2_R, Q_R = cv2.stereoRectify(camera_R_mtx, dist_R, camera_L_mtx, dist_L, size_R, R, T) # for cam R P1 新坐标系下投影阵

print("R1_L  R2_L", R1_L, R2_L)
print("P1_L, P2_L", P1_L, P2_L)
print("R1_R  R2_R", R1_R, R2_R)
print("P1_R, P2_R", P1_R, P2_R)

# calculate undist map
left_map1, left_map2 = cv2.initUndistortRectifyMap(camera_L_mtx, dist_L, R1_L, P1_L, size_L, cv2.CV_16SC2) # for cam L
right_map1, right_map2 = cv2.initUndistortRectifyMap(camera_R_mtx, dist_R, R1_R, P1_R, size_R, cv2.CV_16SC2) # for cam R









#reconstruct
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

cv2.namedWindow("left")
cv2.resizeWindow("left", 800, 800)
cv2.namedWindow("right")
cv2.resizeWindow("right", 800, 800)
cv2.namedWindow("depth")
cv2.resizeWindow("depth", 800, 800)

cv2.moveWindow("left", 0, 0)
cv2.moveWindow("right", 900, 0)
cv2.moveWindow("depth", 0, -900)

cv2.createTrackbar("num", "depth", 0, 10)
cv2.createTrackbar("blocksize", "depth", 5, 255)

def callbackFunc(e,x_pos, y_pos, f, p ): #函数回调
    if e == cv2.EVENT_LBUTTONDOWN:
        print threeD[y][x]
cv2.setMouseCallback("depth", callbackFunc, None)

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0 or not1:
        print("Image capture failure!")
        break
    img_L = cv2.remap(frame0, left_map1,left_map2, cv2.INTER_LINEAR)
    img_R = cv2.remap(frame1, right_map1,right_map2, cv2.INTER_LINEAR)
    img_L_gray =cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
    img_R_gray =cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

    num = cv2.getTrackbarPos("num", "depth")
    blocksize = cv2.getTrackbarPos("blocksize", "depth")
    if blocksize % 2 == 0:
        blocksize +=1
    if blocksize < 5:
        blocksize = 5

    stereo = cv2.StereoBM_create(numDisparities=16*num, blockSize=blockSize)
    disparity = stereo.compute(img_L, img_R) #两幅图像重叠在一起的时候，左摄像机上P的投影和右摄像机上P的投影位置有
                                            #一个距离|Xleft|+|Xright|，这个距离称为Disparity

    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) 

    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., Q_L)

    cv2.imshow("left", img_L)
    cv2.imshow("right",img_R)
    cv2.imshow("depth", disp)

    key == cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord("s"):
        cv2.imwrite("C://Users//MEC 101//Desktop//CAM//Disp//BM_L.jpg", img_L)
        cv2.imwrite("C://Users//MEC 101//Desktop//CAM//Disp//BM_R.jpg", img_R)
        cv2.imwrite("C://Users//MEC 101//Desktop//CAM//Disp//BM_Depth.jpg", disp)

cap0.release()
cap1.release()
cv2.destroyAllWindows()
    

    
            

