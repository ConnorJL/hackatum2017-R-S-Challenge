'''! Dataset reader for the simplified challenge.
'''
from .dataset import record_reader, get_unique_labels
from .record import get_image, has_logo, get_image_part


class SimpleDataset:
    '''! Iterateable object which provieds a simplified dataset.
        Each element is a tupels of (<has logo>, <fixed size numpy array>).
    '''
    class SimpleDatasetIter:
        '''! Iterator for the simple dataset '''
        def __init__(self, record_iter, no_logo_position, label_of_interest):
            self.record_iter = record_iter
            self.no_logo_position = no_logo_position
            self.label_of_interest = label_of_interest

        def __next__(self):
            rec = next(self.record_iter)
            img = get_image(rec.img_path)
            if has_logo(rec):
                for label in rec.labels:
                    if label.name == self.label_of_interest:
                        roi = get_image_part(img, label.pos)
                        return (True, roi, rec.img_path)
            roi = get_image_part(img, self.no_logo_position)
            return (False, roi, rec.img_path)

    def __init__(self, image_path, label_of_interest):
        '''!
        @param image_path           path the the root folder of the dataset
        @param label_of_interest    name of the label to be filtered
        '''
        self.records = record_reader(image_path)
        all_labels = get_unique_labels(self.records)
        assert label_of_interest in all_labels

        position = all_labels[label_of_interest].positions[0]
        size = all_labels[label_of_interest].size
        self.no_logo_position = position + size
        self.label_of_interest = label_of_interest

    def get_logo_size(self):
        ''' The simplified dataset provides fixed size logos. Get this size
        '''
        return self.no_logo_position[2:]

    def has_logo_cnt(self):
        ''' Number of samples with a logo
        '''
        return sum([has_logo(rec) for rec in self.records])

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return SimpleDataset.SimpleDatasetIter(iter(self.records), self.no_logo_position, self.label_of_interest)
