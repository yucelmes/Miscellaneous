
# coding: utf-8

# In[ ]:

import numpy as np
import cv2
from PIL import Image
from math import ceil, radians, sin, cos
import pandas as pd



# Loading Frontal&Profile Face Classifiers
face_cascade1 = cv2.CascadeClassifier(r'.\Downloads\opencv\sources\data\haarcascades_GPU\haarcascade_frontalface_alt.xml')
face_cascade2 = cv2.CascadeClassifier(r'.\Downloads\opencv\sources\data\haarcascades_GPU\haarcascade_profileface.xml')



# Extracting Face Center and Radius
def face_center_radius(x, y, w, h):
    
    face_center = x+w/2, y+h/2
    face_radius_small = np.mean((w, h))/2
    face_radius_big = np.sqrt((w/2)**2 + (h/2)**2)
    
    return face_center, np.int(np.mean((face_radius_small, face_radius_big)))



# Un-Rotating Face Center
def face_original_center(face_center, im_center, angle):
    
    x_dist = face_center[0] - im_center[0]
    y_dist = face_center[1] - im_center[1]
    x_new = x_dist*cos(radians(-angle)) + y_dist*sin(radians(-angle)) + im_center[0]
    y_new = -x_dist*sin(radians(-angle)) + y_dist*cos(radians(-angle)) + im_center[1]
    
    return np.int(x_new), np.int(y_new)



# Creating Mask For Circular Area
def circle_mask(centre, radius, array):
    b, a = centre
    nx, ny = array.shape
    y, x = np.ogrid[-a:nx-a, -b:ny-b]
    mask = x*x + y*y <= radius*radius

    return(mask)



def process_image(img_filenames):
    
    angles = [-30, -15, 0, 15, 30]
    fSF, pSF, MN = 1.043, 1.042, 5

    AnalysisData = pd.DataFrame(columns=['ImageFile', 'BlurredArea', 'CircleNumber', 'AreaPerCircle'])
    i = int(0)    
    
    for img_filename in img_filenames:

        orig_img = Image.open(img_filename)

        # Minimal Background Size Condition: img_h**2 + img_w**2 <= bg_h**2
        img_w, img_h = orig_img.size
        bg_h = np.sqrt(img_h**2 + img_w**2)
        bg_w = bg_h * img_w/img_h
        bg_h, bg_w = np.int(ceil(bg_h)), np.int(ceil(bg_w))
        background = Image.new('RGB', (bg_w, bg_h), (0, 0, 0))
        offset = int((bg_w - img_w) / 2), int((bg_h - img_h) / 2)
        background.paste(orig_img, offset)
        combined_img = np.asarray(background)
        del orig_img

        # Creating 2D Mask for Quantifying False Positives
        face_mask = np.zeros((bg_h, bg_w), 'bool')
        all_radius = []


        for angle in angles:

            img = np.asarray(background.rotate(angle=angle, expand=False)).copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces1 = face_cascade1.detectMultiScale(gray, scaleFactor=fSF, minNeighbors=MN, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(30, 30))
            faces2 = face_cascade2.detectMultiScale(gray, scaleFactor=pSF, minNeighbors=MN, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(30, 30))

            im_center = img.shape[1]/2, img.shape[0]/2


            for (x,y,w,h) in faces1:

                face_center, face_radius = face_center_radius(x, y, w, h)
                new_center = face_original_center(face_center, im_center, angle)
                cv2.circle(combined_img, new_center, face_radius, (255, 0, 0), 3)

                all_radius.append(face_radius)
                face_mask[circle_mask(new_center, face_radius, face_mask)] = True


            for (x,y,w,h) in faces2:

                face_center, face_radius = face_center_radius(x, y, w, h)
                new_center = face_original_center(face_center, im_center, angle)
                cv2.circle(combined_img, new_center, face_radius, (0, 255, 0), 3)

                all_radius.append(face_radius)
                face_mask[circle_mask(new_center, face_radius, face_mask)] = True

    
    
        filename_part = img_filename.split('.')[1:][0]
        Image.fromarray(combined_img[offset[1]:offset[1]+img_h, offset[0]:offset[0]+img_w], 'RGB').save('.' + filename_part + '_P.jpg')  

        blurred_area = face_mask[offset[1]:offset[1]+img_h, offset[0]:offset[0]+img_w].sum()
        fake_mask = np.zeros((bg_h, bg_w), 'bool')
        fake_mask[circle_mask((bg_h/2, bg_w/2), np.sqrt(np.mean(np.power(all_radius, 2))), fake_mask)] = True
        area_per_circle = fake_mask.sum()

        AnalysisData.loc[i] = [img_filename, blurred_area, len(all_radius), area_per_circle]
        i += 1

    return AnalysisData

