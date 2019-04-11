import cv2
import numpy as np
import itertools
import math
import cv2.cv as cv
import glob
import os

"""
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img




img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\attachments\\IMG_1403.JPG")
Img1 = np.zeros(img.shape, np.uint8)
Img2 = img.copy()
Img3 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


cv2.imshow("Image", img)
cv2.imshow("Img1", Img1)
cv2.imshow("Img2", Img2)
cv2.imshow("Img3", Img3)

cv2.imwrite("C:\\Users\\MEC 101\\Desktop\\CAM\\attachments\\IM1.JPG", img,[int(cv2.IMWRITE_JPEG_QUALITY), 5])
cv2.imwrite("C:\\Users\\MEC 101\\Desktop\\CAM\\attachments\\IM2.JPG", img,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
cv2.imwrite("C:\\Users\\MEC 101\\Desktop\\CAM\\attachments\\IMn1.PNG", img,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
cv2.imwrite("C:\\Users\\MEC 101\\Desktop\\CAM\\attachments\\IMn2.PNG", img,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])






cv2.waitKey (0)
cv2.destroyAllWindows()


"""


"""
def salt(img,n):
    '''add pepper noise, n is the number of noise dots'''
    for k in range(n):
        i = int(np.random.random() * img.shape[1]); ''' shape[1] column'''
        j = int(np.random.random() * img.shape[0]); ''' shape[0] rows'''
        if img.ndim == 2:
            img[j,i] = 255
        elif img.ndim == 3:
            img[j,i,0] = 255
            img[j,i,1] = 255
            img[j,i,2] = 255
    return img

if __name__ == '__main__':
    img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\attachments\\IMG_1403.JPG")
    saltImage = salt( img,700)
    cv2.imshow("Salt", saltImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""



"""
img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\attachments\\IMG_1403.JPG")
'''
b,g,r= cv2.split(img)
cv2.imshow('Blue', r)
cv2.imshow('Green', b)
cv2.imshow('Red', g)
'''
b = np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)
g = np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)
r = np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)


b[:,:] = img[:,:,0]
g[:,:] = img[:,:,1]
r[:,:] = img[:,:,2]


merged = cv2.merge([b,g,r])
print merged.strides
cv2.imshow("Merged", merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""





"""
def calcAndDrawHist(img, color):
    hist = cv2.calcHist([img],[0],None,[256],[0.0,256.0])
    minV, maxV, minLoca, maxLoca = cv2.minMaxLoc(hist)
    histImg = np.zeros([256,256,3], np.uint8)
    hpt = int(0.9 *256);

    for h in range(256):
        intensity = int(hist[h]*hpt/maxV)
        cv2.line(histImg,(h,256),(h,256-intensity),color)

    return histImg;
if __name__ == '__main__':
    img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\118\\Laserdot\\DSC_0030.JPG")
    b,g,r = cv2.split(img)

    histImgB = calcAndDrawHist(b,[255,0,0])
    histImgG = calcAndDrawHist(g,[0,255,0])
    histImgR = calcAndDrawHist(r,[0,0,255])

    cv2.imshow("histB", histImgB)
    cv2.imshow("histG", histImgG)
    cv2.imshow("histR", histImgR)
    #cv2.imshow("Img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""

"""
img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\attachments\\IMG_1403.JPG",0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(10,10))
eroded = cv2.erode(img,element)
cv2.imshow("eroded Img", eroded);

dilated = cv2.dilate(img,element)
cv2.imshow("dialeted Img", dilated);

cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\attachments\\IMG_1403.JPG",0)
element = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, element)
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
cv2.imshow("Closed", closed);
cv2.imshow("Opened", opened);
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\attachments\\boss.JPG")

element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
dilated = cv2.dilate(img, element)
eroded = cv2.erode(img,element)

result = cv2.absdiff(dilated, eroded);
retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY);
result = cv2.bitwise_not(result);
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\DSC_0005.JPG")
imgGRAY = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\DSC_0005.JPG",0)
cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
lin = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

lin[0,0] = 0
lin[0,1] = 0
lin[1,0] = 0
lin[0,3] = 0
lin[0,4] = 0
lin[1,4] = 0
lin[3,0] = 0
lin[4,0] = 0
lin[4,1] = 0
lin[3,4] = 0
lin[4,4] = 0
lin[4,3] = 0

square = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
cha = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

resultA = cv2.dilate(imgGRAY,cross)
resultA = cv2.erode(resultA,lin)
resultB = cv2.dilate(imgGRAY,cha)
resultB = cv2.erode(resultB,square)

result = cv2.absdiff(resultB,resultA)
retval,result = cv2.threshold(result,40,255,cv2.THRESH_BINARY)
for j in range(result.size):
    y = j/result.shape[0]
    x = j%result.shape[0]

    if result[x,y] == 255:
        cv2.circle(imgGRAY,(y,x),5,(255,0,0))

cv2.namedWindow("Img", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Img", 600,600)
cv2.imshow("Img", imgGRAY)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""


img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\attachments\\boss.JPG")
dst =  cv2.boxFilter(img,-1,(5,5))
cv2.imshow("blurr", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
def salt(img, n):  
    for k in range(n):  
        i = int(np.random.random() * img.shape[1]);  
        j = int(np.random.random() * img.shape[0]);  
        if img.ndim == 2:   
            img[j,i] = 255  
        elif img.ndim == 3:   
            img[j,i,0]= 255  
            img[j,i,1]= 255  
            img[j,i,2]= 255
    return img

img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\attachments\\boss.JPG")
result = salt(img,800)
median = cv2.medianBlur(result,3)

cv2.imshow("Salt",result)
cv2.imshow("Median",median)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\attachments\\boss.JPG")

x = cv2.Sobel(img,cv2.CV_16S,1,0)
y = cv2.Sobel(img,cv2.CV_16S,0,1)
absX = cv2.convertScaleAbs(x)  
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
 
cv2.imshow("absX", absX)
cv2.imshow("absY", absY)
 
cv2.imshow("Result", dst)
 
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\DSC_0005.JPG")

lap = cv2.Laplacian(img, cv2.CV_16S,ksize = 5)
dst = cv2.convertScaleAbs(lap)
cv2.namedWindow("Img", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Img", 600,600)
cv2.imshow("Img", dst) 
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
"""
img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\DSC_0005.JPG",0)
img = cv2.GaussianBlur(img, (3,3),0)
canny = cv2.Canny(img, 1, 50)

cv2.namedWindow("Img", cv2.WINDOW_NORMAL)
cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Img", 600,600)
cv2.resizeWindow("Canny", 600,600)
cv2.imshow("Img", img)
cv2.imshow("Canny", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
"""
img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\DSC_0005.JPG",0)
img = cv2.GaussianBlur(img,(3,3),0)
edge = cv2.Canny(img, 10, 50, 3)
lines = cv2.HoughLines(edge,1,np.pi/180,118)
result = img.copy()
for line in lines[0]:
    rho = line[0]
    theta = line[1]
    print rho
    print theta
    if (theta < (np.pi/4.0)) or (theta >(3.0*np.pi/4.0)):
        pt1 = (int(rho/np.cos(theta)),0)
        pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0])
        cv2.line( result, pt1, pt2, (255))
    else:
        pt1 = (0,int(rho/np.sin(theta)))
        pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
        cv2.line(result, pt1, pt2, (255), 1)
    
cv2.namedWindow("Img", cv2.WINDOW_NORMAL)
cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Img", 600,600)
cv2.resizeWindow("Canny", 600,600)
cv2.imshow("Img", result)
cv2.imshow("Canny", edge)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\DSC_0005.JPG")
img = cv2.GaussianBlur(img,(3,3),0)
edge = cv2.Canny(img, 50, 150, 3)
lines = cv2.HoughLines(edge,1,np.pi/180,118)
result = img.copy()

lines = cv2.HoughLinesP(edge, 1, np.pi/180, 80, 10, 7)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    
cv2.namedWindow("Img", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Img", 600,600)
cv2.imshow("Img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\DSC_0005.JPG",0)
lut = np.zeros(256, dtype = img.dtype)
hist= cv2.calcHist([img],
                   [0],
                   None,
                   [256],
                   [0.0,255.0])
minPnum, maxPnum = 0, 255

for Pnum,Pval in enumerate(hist):
    if Pval != 0:
        minPnum = Pnum
        break
for Pnum,Pval in enumerate(reversed(hist)):
    if Pval != 0:
        maxPnum = 255-Pnum
        break
print minPnum, maxPnum

for i,v in enumerate(lut):
    print i
    if i < minPnum:
        lut[i] = 0
    elif i >maxPnum:
        lut[i] = 255
    else:
        lut[i] = int(255.0*(i-minPnum)/(maxPnum-minPnum)+0.5)

 
result = cv2.LUT(img,lut)
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.namedWindow("Lut", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result", 600,600)
cv2.resizeWindow("Lut", 600,600)
cv2.imshow("Result", result)
cv2.imwrite("Lut", result)
cv2.waitKey(0)
cv2.destroyAllWindows()   
 """
"""
img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\DSC_0005.JPG")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 125,255,cv2.THRESH_BINARY)
cont, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,cont,-1,(0,255,255),3)
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result", 600,600)
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
import cv2.cv as cv
img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\118\\Laserdot\\L_120min_2.JPG")
IMG = cv.CreateImage((img.shape[1],img.shape[0]),8,3)
IMG = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

lower_red = np.array([0,100,100])
upper_red = np.array([8,255,255])
mask0 = cv2.inRange(IMG, lower_red, upper_red)

lower_red = np.array([170,100,100])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(IMG, lower_red, upper_red)

mask = mask0+mask1

output_img = img.copy()
output_img[np.where(mask==0)] = 255
output_img[np.where(mask!=0)] = 0

def calcAndDrawHist(img, color):
    hist = cv2.calcHist([img],[0],mask,[256],[0.0,256.0])
    minV, maxV, minLoca, maxLoca = cv2.minMaxLoc(hist)
    histImg = np.zeros([256,256,3], np.uint8)
    hpt = int(0.9 *256);

    for h in range(256):
        intensity = int(hist[h]*hpt/maxV)
        cv2.line(histImg,(h,256),(h,256-intensity),color)

    return histImg;

b,g,r =  cv2.split(img)
histImgR = calcAndDrawHist(r,[0,0,255])
histImgG = calcAndDrawHist(g,[0,255,0])
histImgB = calcAndDrawHist(b,[255,0,0])

cv2.imshow("Red",histImgR)
cv2.imshow("Gre",histImgG)
cv2.imshow("Blu",histImgB)
cv2.waitKey(0)    
cv2.destroyAllWindows() 
"""
"""
lower_val = 10
higher_val = 100
height, width, _  = img.shape
filter_img =  np.zeros(img.shape, np.uint8)
height, width = mask.shape
for x in range(height):
    for y in range(width):
        b, g, r = img[x][y]
        if b < lower_val and g < lower_val and r>= higher_val:
            filter_img[x][y] = 255
cv2.imwrite("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\red_Filter\\Filter_img.JPG", filter_img)

class Dot:
    def __init__(self):
        self.pixels = set()
        self.center = None
    def forceAdd(self, x,y):
        self.pixels.add((x,y))
    def add(self, x, y):
        if any((x + a, y + b) in self.pixels for a, b in itertools.product([-2, -1, 0, 1, 2], repeat = 2)):
            self.pixels.add((x, y))
            return True
        return False
    def merge(self,dot):
        self.pixels = self.pixels.union(dot.pixels)
    def size(self):
        return len(self.pixels)

filter_img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\red_Filter\\Filter_img.JPG", cv2.IMREAD_GRAYSCALE)

filter_pixels = []
height, width, _  = img.shape
for x in range(height):
    for y in range(width):
        if filter_img[x][y] > 200:
            filter_pixels.append((x,y))
dots = []
for x, y in filter_pixels:
    this_dots = [d for d in dots if d.add(x, y)]

    if len(this_dots) == 0:
        new_dot = Dot()
        new_dot.forceAdd(x,y)
        dots.append(new_dot)
    if len(this_dots) > 1:
        this_dot = this_dots[0]
    for dot in this_dots[1:]:
            this_dot.merge(dot)
            dots.remove(dot)
for dot in dots:
    bright_value = None
    bright_pixel = None
    for x,y in dot.pixels:
        v = math.sqrt(sum(pow(v,2.0) for v in img[x][y]))
        if bright_value == None or bright_value < v:
            bright_value = v
            bright_pixel = (x,y)
    dot.center = bright_pixel
    
img = cv2.circle(img, dot.center, 5, (0,255,255), 3)

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.resizeWindow("img", 600,600)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*6,3),np.float32)
objp[:,:2] = np.mgrid[0:6,0:6].T.reshape(-1,2)


obj_points = [] #3D points
img_points = [] #2D points
"""
"""
img = glob.glob("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\118\\Laserdot\\*.jpg")
img_save = "C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\118\\Laserdot\\Filter"

for fname in img:
    Img = cv2.imread(fname)
    IMG = cv.CreateImage((Img.shape[1],Img.shape[0]),8,3)
    IMG = cv2.cvtColor(Img,cv2.COLOR_BGR2HSV)
    (img_dir, tempfilename) = os.path.split(fname) # 分离文件目录 文件名和文件后缀
    # Apply filter
    lower_red = np.array([0,50,50]) 
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(IMG, lower_red, upper_red)

    lower_red = np.array([160,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(IMG, lower_red, upper_red)

    mask = mask0+mask1

    output_img = Img.copy()
    output_img[np.where(mask==0)] = 0
    output_img[np.where(mask!=0)] = 255
    #save path
    savepath = os.path.join(img_save, tempfilename)
    cv2.imwrite(savepath, output_img)
"""
"""
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)    
img = glob.glob("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\V\\*.jpg")
"""
"""
objp = np.zeros((6*8,3),np.float32)
objp[:,:2] = np.mgrid[0:6,0:8].T.reshape(-1,2)

obj_points = [] #3D points
img_points = [] #2D points

for fname in img:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2. findChessboardCorners(gray, (6,8), None)

    if ret == True:
        cv2.cornerSubPix(gray, corners, (11,11),(-1,-1),criteria)
        obj_points.append(objp)
        img_points.append(corners)


        cv2.drawChessboardCorners(img, (6,8),corners,ret)
        cv2.namedWindow("FindCorners", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("FindCorners", 600,600)
        cv2.imshow("FindCorners", img)
        cv2.waitKey(0)
cv2.destroyAllWindows()

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
"""
"""
"""
"""
#project points and write coordinate on image

axis = np.float32([[3,0,0],[0,3,0],[0,0,-3]]).reshape(-1,3)


for fname in glob.glob("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\118\\Checkboard\\*.jpg"):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2. findChessboardCorners(gray, (6,6), None)

    if ret == True:
        Rvecs, Tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)
        img_pts, _ = cv2.projectPoints(axis, Rvecs, Tvecs, mtx, dist)
        draw(img,corners,img_pts)
        """
"""
        corners2 = np.int0(corners)
    for i in corners2:
        x,y = i.ravel()
        cv2.circle(img,(x,y),5,(0,255,255),5)
        """
"""
    cv2.namedWindow("FKimg", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FKimg", 600,600)
    cv2.imshow('FKimg',img)
    cv2.waitKey(0)


cv2.destroyAllWindows()



Mrot,_ = cv2.Rodrigues(Rvecs)
print " Mrot ",Mrot

#undistort

img_dist = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\118\\Checkboard\\DSC_0002.jpg")
h,w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist,(6,6),1,(6,6))
dist = cv2.undistort(img_dist, mtx, dist, None, newcameramtx)
x,y,c,d = roi
cv2.imwrite("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\118\\Checkboard\\dist.jpg",dist)

"""
"""

hls = cv2.cvtColor(output_img, cv2.COLOR_BGR2HLS)
for i in range(1,100):
    ind = (hls[:,:,1] > 0) & (hls[:,:,1] < (220-i*2))
    hls[:,:,1] += ind
bgr = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)


"""
"""
img = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\118\\Laserdot\\DSC_0010.jpg")
height, width,_ = img.shape
filter_img = np.zeros((height,width),np.uint8)
low_val = 10
high_val = 100
for x in range(height):
    for y in range(width):
        b,g,r = img[x][y]
        if b <= low_val and g <= low_val and r >=high_val:
            filter_img[x][y] = 255 # now we filtered out none red dots
cv2.namedWindow("Dots", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Dots", 600,600)
cv2.imshow('Dots',filter_img)
cv2.waitKey(0)
cv2. destroyAllWindows()
cv2.imwrite("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\118\\Laserdot\\filter_img.jpg", filter_img)
"""
# find center dot
"""
class Dot:
    def __init__(self):
        self.pixels = set()
        self.center = None

        self.coord = None
        self.gm_vals = None
    def forceAdd(self, x, y):
        self.pixels.add((x, y))
    def add(self, x, y):
        if any((x + a, y + b) in self.pixels for a, b in itertools.product([-2, -1, 0, 1, 2], repeat = 2)):
            self.pixels.add((x, y))
            return True
        return False
    def merge(self, dot):
        self.pixels = self.pixels.union(dot.pixels)
        
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

imgc = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\118\\Laserdot\\DSC_0010.jpg", cv2.IMREAD_GRAYSCALE) 
filtered_pixels = []
height, width, = imgc.shape
for x in range(height):
    for y in range(width):
        if imgc[x][y] > 50:
            filtered_pixels.append((x,y))

# find connected dots
dots = []
for x, y in filtered_pixels:

    this_dots = [d for d in dots if d.add(x, y)]

    if len(this_dots) == 0:
        new_dot = Dot()
        new_dot.forceAdd(x,y)
        dots.append(new_dot)
    if len(this_dots) == 1:
        pass
    if len(this_dots) >1:
        this_dots = this_dots[0]
        for dot in this_dots[1:]:
            this_dots.merge(dots)
            dots.remove(dot)
#find center of each dots
for dot in dots:
    bright_value = None
    bright_pixel = None

    for x, y in dot.pixels:
        v = math.sqrt(sum(pow(v, 2.0) for v in dot_img[x][y]))
        if bright_value == None or bright_value < v:
            bright_value = v
            bright_pixel = (x, y)

    dot.center = map(float, bright_pixel)
"""
"""
img1 = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\V\\L_0min_1.jpg")
img2 = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\V\\L_0min_2.jpg")
img3 = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\V\\L_0min_3.jpg")
img4 = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\V\\L_30min_1.jpg")
img5 = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\V\\L_30min_2.jpg")
img6 = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\V\\L_30min_3.jpg")
img7 = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\V\\L_60min_1.jpg")
img8 = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\V\\L_60min_2.jpg")
img9 = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\V\\L_60min_3.jpg")
img10 = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\V\\L_90min_1.jpg")
img11 = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\V\\L_90min_2.jpg")
img12 = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\V\\L_90min_3.jpg")
img13 = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\V\\L_120min_1.jpg")
img14 = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\V\\L_120min_2.jpg")
img15 = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\V\\L_120min_3.jpg")
res0 = cv2.add(img1,img2)
res1 = cv2.add(res0, img3)
res2 = cv2.add(res1, img4)
res3 = cv2.add(res2, img5)
res4 = cv2.add(res3, img6)
res5 = cv2.add(res4, img7)
res6 = cv2.add(res5, img8)
res7 = cv2.add(res6, img9)
res8 = cv2.add(res7, img10)
res9 = cv2.add(res8, img11)
res10 = cv2.add(res9, img12)
res11 = cv2.add(res10, img13)
res12 = cv2.add(res11, img14)

res = cv2.add(res12, img15)
cv2.namedWindow("Dots", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Dots", 600,600)
cv2.imshow('Dots', res3)
cv2.waitKey(0)
cv2. destroyAllWindows()

"""

img1 = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\V\\L_0min_1.jpg")
img2 = cv2.imread("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\V\\L_30min_2.jpg")
#res_add = cv2.add(img1,img2)
img2gray = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)#get 1st img "dark"
lower_red = np.array([0,50,50]) 
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(img2gray, lower_red, upper_red)

lower_red = np.array([160,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img2gray, lower_red, upper_red)
mask = mask0 + mask1


img1_d = img1.copy()
img1_d[np.where(mask==0)] = 0 # this filter out non red part in img1




#bitwise_and 取交集
height, width, channels = img2.shape
roi = img1_d[0:height,0:width]# 截取要改的那一部分

img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
lower_red = np.array([0,50,50]) 
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(img2gray, lower_red, upper_red)

lower_red = np.array([160,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img2gray, lower_red, upper_red)
mask = mask0 + mask1

img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)#turn gray
#ret,mask = cv2.threshold(img2gray, 10,255,cv2.THRESH_BINARY)#二值化以提取图像中的元素

mask_inv = cv2.bitwise_not(mask)
img1bg = cv2.bitwise_and(roi,roi, mask=mask_inv)#截出洞
img2bg = cv2.bitwise_and(img2, img2, mask=mask)#截出点元素
res = cv2.add(img1bg,img2bg)

img1_d[0:height,0:width] = res


cv2.namedWindow("Dots", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Dots", 600,600)
cv2.imshow('Dots', img1_d)
cv2.waitKey(0)
cv2. destroyAllWindows()

cv2.imwrite("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\V\\Result\\diff_30min.JPG", img1_d)
"""
"""                  

        
        
        
    
    
            


        
"""


cv2.imwrite("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\red_Filter\\DSC_0002.JPG", output_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
cv2.imwrite("C:\\Users\\MEC 101\\Desktop\\CAM\\1-11\\100NCD50\\red_Filter_Bri\\0002.JPG", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])



cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.namedWindow("Output_img", cv2.WINDOW_NORMAL)
cv2.namedWindow("Bri", cv2.WINDOW_NORMAL)
cv2.resizeWindow("img", 600,600)
cv2.resizeWindow("Output_img", 600,600)
cv2.resizeWindow("Bri", 600,600)
cv2.imshow("img", img)
cv2.imshow("Output_img", output_img)
cv2.imshow("Bri",mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

        
    


