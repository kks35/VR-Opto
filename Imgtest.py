import cv2

img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\attachments\\IMG_1403.JPG")
cv2.namedWindow("Image")
cv2.imshow("Image", img)
cv2.waitKey (0)
