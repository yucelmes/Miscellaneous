# In[ ]:

# Parameter Optimization of OpenCV Face Detection by Haar-Cascade

import numpy as np
import cv2
from PIL import Image
from math import ceil, radians, sin, cos


# In[ ]:

# Loading Frontal&Profile Face Classifiers

face_cascade1 = cv2.CascadeClassifier(r'.\opencv\sources\data\haarcascades_GPU\haarcascade_frontalface_alt.xml')
face_cascade2 = cv2.CascadeClassifier(r'.\opencv\sources\data\haarcascades_GPU\haarcascade_profileface.xml')


# In[ ]:

# Extracting Face Center and Radius

def face_center_radius(x, y, w, h):
    
    face_center = x+w/2, y+h/2
    face_radius_small = np.mean((w, h))/2
    face_radius_big = np.sqrt((w/2)**2 + (h/2)**2)
    
    return face_center, np.int(np.mean((face_radius_small, face_radius_big)))


# In[ ]:

# Un-Rotating Face Center

def face_original_center(face_center, im_center, angle):
    
    x_dist = face_center[0] - im_center[0]
    y_dist = face_center[1] - im_center[1]
    x_new = x_dist*cos(radians(-angle)) + y_dist*sin(radians(-angle)) + im_center[0]
    y_new = -x_dist*sin(radians(-angle)) + y_dist*cos(radians(-angle)) + im_center[1]
    
    return np.int(x_new), np.int(y_new)


# In[ ]:

# Creating Mask For Rectangular Area

def rectangular_blur(final_faces, img_array):
    
    large_faces = [face for face in final_faces if 2*face[1] > 0.25*min(img_array.shape[:2])]
    small_faces = [face for face in final_faces if 2*face[1] <= 0.25*min(img_array.shape[:2])]
    
    temp_img_array = img_array.copy()
    blur_weak = [41, 39, 35, 27]
    blur_strong = [81, 75, 69, 63, 57, 49, 41, 33]
    percents_weak = np.array([0.5, 0.6, 0.8, 1])
    percents_strong = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            
    
    masks = [np.zeros(img_array.shape[:2], 'bool') for _ in range(len(percents_weak))]        
    for center, radius in small_faces:
        rads = (radius * percents_weak).astype(int)
        for mask_no, rad in enumerate(rads):
            x1 = max(0, center[1]-rad)
            y1 = max(0, center[0]-rad)
            masks[mask_no][x1:center[1]+rad, y1:center[0]+rad] = True                         
    edge_masks = [masks[0]]
    for mask_no in range(len(masks)-1):
        edge_masks.append(masks[mask_no+1] ^ masks[mask_no])    
    for mask, strength in zip(edge_masks, blur_weak):
        temp_img_array[mask] = cv2.medianBlur(img_array, strength)[mask]
            
            
    for center, radius in large_faces:
        rads = (radius * percents_strong).astype(int)
        masks = [np.zeros(img_array.shape[:2], 'bool') for _ in range(len(percents_strong))]
        for mask_no, rad in enumerate(rads):
            x1 = max(0, center[1]-rad)
            y1 = max(0, center[0]-rad)
            masks[mask_no][center[1]-rad:center[1]+rad, center[0]-rad:center[0]+rad] = True
        edge_masks = [masks[0]]
        for mask_no in range(len(masks)-1):
            edge_masks.append(masks[mask_no+1] ^ masks[mask_no])    
        for mask, strength in zip(edge_masks, blur_strong):
            temp_img_array[mask] = cv2.medianBlur(img_array, strength)[mask]
    
    
    return temp_img_array


# In[ ]:

# Frontal and Profile Face Detection

def process_image(img_filenames):
    
    abs_angle = 30
    angles = np.linspace(-abs_angle, abs_angle, 5)
    
    # Algorithmic Parameters
    fSF, pSF, MN = 1.02, 1.035, 5


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

        # Face Centers and Radiuses
        final_faces = []


        for angle in angles:

            img = np.asarray(background.rotate(angle=angle, expand=False)).copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces1 = face_cascade1.detectMultiScale(gray, scaleFactor=fSF, minNeighbors=MN, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(30, 30))
            faces2 = face_cascade2.detectMultiScale(gray, scaleFactor=pSF, minNeighbors=MN, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(30, 30))
            im_center = img.shape[1]/2, img.shape[0]/2



            # Frontal Face Analysis
            for (x,y,w,h) in faces1:

                face_center, face_radius = face_center_radius(x, y, w, h)
                if 2*face_radius <= 0.5*img_h:          # Filter 1: 50% Size Threshold
                    new_center = face_original_center(face_center, im_center, angle)
                    new_center = new_center[0] - offset[0], new_center[1] - offset[1]
                    final_faces.append((new_center, face_radius))


            # Profile Face Analysis
            for (x,y,w,h) in faces2:

                face_center, face_radius = face_center_radius(x, y, w, h)
                if 2*face_radius <= 0.5*img_h:          # Filter 1: 50% Size Threshold
                    new_center = face_original_center(face_center, im_center, angle)
                    new_center = new_center[0] - offset[0], new_center[1] - offset[1]
                    final_faces.append((new_center, face_radius))


        combined_img = combined_img[offset[1]:offset[1]+img_h, offset[0]:offset[0]+img_w]
        combined_img = rectangular_blur(final_faces, combined_img)    
        
        filename_part = img_filename.split('.')[0]
        Image.fromarray(combined_img, 'RGB').save(filename_part + '_FP.jpg') 


   

    return None
