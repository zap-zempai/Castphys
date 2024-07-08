import numpy as np
import cv2
from pathlib import Path

# GLOBAL PARAMS ------------------------------------------------------------
MIN_MATCH_COUNT = 7
FLANN_INDEX_KDTREE = 1

### FUNCIONS ------------------------------------------------------------------
def search_light(img_q,img_t):
        # Initiate SIFT detector and flann
    sift = cv2.SIFT_create()
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # find the keypoints and descriptors with SIFT
    kp_q, des_q = sift.detectAndCompute(img_q,None)
    kp_t, des_t = sift.detectAndCompute(img_t,None)
    # Comput match
    matches = flann.knnMatch(des_q,des_t,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        ## all
        src_pts = np.float32([ kp_q[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_t[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        #matchesMask = mask.ravel().tolist()
        h,w = img_q.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        #print(dst)
        # limits
        x0 = int(dst[:,:,0].min())
        y1 = int(dst[:,:,1].min())
        x1 = x0 + int((dst[:,:,0].max() - x0) * 0.5)
        y0 = y1 - int((dst[:,:,1].max() - y1) * 0.7)

    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        x0,y0,x1,y1 = -1,-1,-1,-1

    return x0,y0,x1,y1


def search_box(path_img, path_mark, look_box=False):
    # Load Images
    img_mark = cv2.imread(str(path_mark), cv2.IMREAD_GRAYSCALE) # queryImage
    img_light = cv2.imread(str(path_img), cv2.IMREAD_GRAYSCALE) # trainImage

    x0,y0,x1,y1 = search_light(img_mark,img_light)

    if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
        raise Exception(f"Error: Light not found")
    
    # lock img
    if look_box:
        img = cv2.imread(str(path_img))
        cv2.rectangle(img,(x0,y0),(x1,y1),(0,255,0),1)
        cv2.imshow('imagen',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return x0,y0,x1,y1
