# In[ ]:

import collections
import math

import cv2
import numpy
#from PIL import Image
import tensorflow as tf

import common
import model


# In[ ]:

def make_scaled_ims(im, min_shape):
    ratio = 1. / 2 ** 0.5
    shape = (im.shape[0], im.shape[1])

    while True:
        shape = (int(shape[0] * ratio), int(shape[1] * ratio))
        if shape[0] < min_shape[0] or shape[1] < min_shape[1]:
            break
        yield cv2.resize(im, (shape[1], shape[0]))


# In[ ]:

def detect(im, param_vals):
    """
    Detect number plates in an image.

    :param im:
        Image to detect number plates in.

    :param param_vals:
        Model parameters to use. These are the parameters output by the `train`
        module.

    :returns:
        Iterable of `bbox_tl, bbox_br, letter_probs`, defining the bounding box
        top-left and bottom-right corners respectively, and a 7,36 matrix
        giving the probability distributions of each letter.

    """
    
    # Convert the image to various scales.
    scaled_ims = list(make_scaled_ims(im, model.WINDOW_SHAPE))
    
    # Load the model which detects number plates over a sliding window.
    x, y, params = model.get_detect_model()
    
    # Execute the model at each scale.
    with tf.Session(config=tf.ConfigProto()) as sess:
        y_vals = []
        for scaled_im in scaled_ims:
            feed_dict = {x: numpy.stack([scaled_im])}
            feed_dict.update(dict(zip(params, param_vals)))
            y_vals.append(sess.run(y, feed_dict=feed_dict))
    
    # Interpret the results in terms of bounding boxes in the input image.
    # Do this by identifying windows (at all scales) where the model predicts a
    # number plate has a greater than 50% probability of appearing.
    #
    # To obtain pixel coordinates, the window coordinates are scaled according
    # to the stride size, and pixel coordinates.
    
    for i, (scaled_im, y_val) in enumerate(zip(scaled_ims, y_vals)):
        
        for window_coords in numpy.argwhere(y_val[0, :, :, 0] > -math.log(1./0.60 - 1)):   #### 0.99 -> 0.6 ###
            
            letter_probs = (y_val[0,
                                  window_coords[0],
                                  window_coords[1], 1:].reshape(
                                    7, len(common.CHARS)))
            letter_probs = common.softmax(letter_probs)

            img_scale = float(im.shape[0]) / scaled_im.shape[0]

            bbox_tl = window_coords * (8, 4) * img_scale
            bbox_size = numpy.array(model.WINDOW_SHAPE) * img_scale

            present_prob = common.sigmoid(
                               y_val[0, window_coords[0], window_coords[1], 0])
            
            yield bbox_tl, bbox_tl + bbox_size, present_prob, letter_probs


# In[ ]:

def _overlaps(match1, match2):
    bbox_tl1, bbox_br1, _, _ = match1
    bbox_tl2, bbox_br2, _, _ = match2
    return (bbox_br1[0] > bbox_tl2[0] and
            bbox_br2[0] > bbox_tl1[0] and
            bbox_br1[1] > bbox_tl2[1] and
            bbox_br2[1] > bbox_tl1[1])


# In[ ]:

def _group_overlapping_rectangles(matches):
    matches = list(matches)
    num_groups = 0
    match_to_group = {}
    for idx1 in range(len(matches)):
        for idx2 in range(idx1):
            if _overlaps(matches[idx1], matches[idx2]):
                match_to_group[idx1] = match_to_group[idx2]
                break
        else:
            match_to_group[idx1] = num_groups 
            num_groups += 1

    groups = collections.defaultdict(list)
    for idx, group in match_to_group.items():
        groups[group].append(matches[idx])

    return groups


# In[ ]:

def post_process(matches):
    """
    Take an iterable of matches as returned by `detect` and merge duplicates.

    Merging consists of two steps:
      - Finding sets of overlapping rectangles.
      - Finding the intersection of those sets, along with the code
        corresponding with the rectangle with the highest presence parameter.

    """
    groups = _group_overlapping_rectangles(matches)

    for group_matches in groups.values():
        mins = numpy.stack(numpy.array(m[0]) for m in group_matches)
        maxs = numpy.stack(numpy.array(m[1]) for m in group_matches)
        present_probs = numpy.array([m[2] for m in group_matches])
        letter_probs = numpy.stack(m[3] for m in group_matches)

        yield (numpy.max(mins, axis=0).flatten(),
               numpy.min(maxs, axis=0).flatten(),
               numpy.max(present_probs),
               letter_probs[numpy.argmax(present_probs)])


# In[ ]:

def letter_probs_to_code(letter_probs):
    return "".join(common.CHARS[i] for i in numpy.argmax(letter_probs, axis=1))


# In[ ]:

# Creating Mask For Rectangular Area

def rectangular_blur(final_plates, img_array):
    
    blur_strength = [17, 21, 25, 31, 39]
    percents = [0.1, 0.2, 0.3, 0.4]
    
    for x_range, y_range in final_plates:
        
        x_ranges = [x_range]
        y_ranges = [y_range]
        
        # Outer Frames
        for percent in percents:
            
            x_length = int(percent*(x_range[1]-x_range[0]))
            y_length = int(percent*(y_range[1]-y_range[0]))
            
            x_ranges.append((x_range[0]+x_length, x_range[1]-x_length))
            y_ranges.append((y_range[0]+y_length, y_range[1]-y_length))
            
             
        for x, y, strength in zip(x_ranges, y_ranges, blur_strength):
            
            plate_part = img_array[y[0]:y[1], x[0]:x[1]]
            plate_part = cv2.medianBlur(plate_part, strength)
            
            img_array[y[0]:y[0]+plate_part.shape[0], x[0]:x[0]+plate_part.shape[1]] = plate_part
        
                    
    return img_array



# In[ ]:

def process_image(img_filenames):
    
    f = numpy.load(r'.\weights100.npz')
    param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]

    lower_y = numpy.array([0, 50, 70], dtype='uint8')
    upper_y = numpy.array([80, 190, 200], dtype='uint8')
    lower_w = numpy.array([70, 70, 60], dtype='uint8')
    upper_w = numpy.array([255, 255, 255], dtype='uint8')


    for img_filename in img_filenames:

        im = cv2.imread(img_filename)

        mask_y = cv2.inRange(im, lower_y, upper_y)
        im_y = cv2.bitwise_and(im, im, mask=mask_y)   
        mask_w = cv2.inRange(im, lower_w, upper_w)
        im_w = cv2.bitwise_and(im, im, mask=mask_w)
        ims = [im, im_y, im_w]
        final_plates = []


        for img in ims:

            im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.

            for pt1, pt2, present_prob, letter_probs in post_process(detect(im_gray, param_vals)):

                x_length = abs(pt2[0]-pt1[0])
                y_length = abs(pt2[1]-pt1[1])

                if (35<x_length<750) & (35<y_length<750):

                    pt1 = tuple(reversed(list(map(int, pt1))))
                    pt2 = tuple(reversed(list(map(int, pt2))))

                    x_range = [pt1[0], pt2[0]]; x_range.sort()
                    y_range = [pt1[1], pt2[1]]; y_range.sort()

                    final_plates.append((x_range, y_range))


        im = rectangular_blur(final_plates, im)
        filename_part = img_filename.split('.')[0]
        # Image.fromarray(im, 'RGB').save(filename_part + '_NP.jpg')
        cv2.imwrite(filename_part + '_NP.jpg', im)
    
    
    
    
    return None
