'''!
    Interface to the dataset
'''

class Label(object):
    '''!
        Label describing a single logo in a picture
    '''
    def __init__(self, name, pos):
        ## string naming a category
        self.name = name
        ## Bounding box [x1, y1, x size, y size].
        ## (x1, y1) are the coordinates of the bounding box' top left corner
        self.pos = pos

class Record(object):
    '''!
        Description of a single picture including logos
    '''
    def __init__(self, img_path, labels):
        ## image path relative to dataset root
        self.img_path = img_path
        ## array of 'Label' objects
        self.labels = labels

