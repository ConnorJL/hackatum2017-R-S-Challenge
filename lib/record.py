'''!
Get information or data of a single record
'''
import skimage.io as io
from .dataset_interface import Record

def has_logo(rec):
    ''' returns true if there is a logo in the record '''
    return len(rec.labels) > 0

def get_image(img_path):
    ''' returns numpy array with complete RGB image data '''
    return io.imread(img_path)

def get_image_part(img, pos):
    ''' get image data for a given position p, with p as [x1, y1, xsize, ysize] '''
    return img[pos[1]:(pos[1]+pos[3]), pos[0]:(pos[0]+pos[2])]

def get_image_corner(img, corner, corner_size=[90, 140]):
    ''' Get a 90 x 140 slice from the corner of the image.
    The corner of interest is defined by the argument corner, with
    (True, True) as top left
    (True, False) as top right
    (False, True) as bottom left
    (False, False) as bottom right '''
    if (True, True) == corner:
        return img[:corner_size[0], :corner_size[1]]
    if (True, False) == corner:
        return img[:corner_size[0], -corner_size[1]:]
    if (False, True) == corner:
        return img[-corner_size[0]:, :corner_size[1]]
    if (True, True) == corner:
        return img[-corner_size[0]:, -corner_size[1]:]

def bounding_box_label(img, labels):
    ''' draw a bounding box around all given labels '''
    for label in labels:
        p = label.pos
        color = [0xff, 0, 0]
        border = 3
        img[p[1]:(p[1]+border), p[0]:(p[0]+p[2])] = color
        img[p[1]:(p[1]+p[3]), p[0]:(p[0]+border)] = color
        img[(p[1]+p[3]-border):(p[1]+p[3]), p[0]:(p[0]+p[2])] = color
        img[p[1]:(p[1]+p[3]), (p[0]+p[2]-border):(p[0]+p[2])] = color
    return img
