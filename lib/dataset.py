'''!
Methods to load all records from a dataset, including metadata
'''

import csv
import os
from .dataset_interface import Label, Record

LABELS_FILE = "metadata.txt"

def read_labels(labels_file_path):
    '''!
    parse a label file
    '''
    if os.path.exists(labels_file_path):
        with open(labels_file_path, newline='') as labels_file:
            reader = csv.reader(labels_file, delimiter=',')
            all_labels = []
            for row in reader:
                pos = [int(x) for x in row[1:]]
                assert len(pos) == 4, "invalid label %s" % labels_file_path
                all_labels.append(Label(row[0], pos))
            return all_labels
    else:
        return []

def record_reader(images_path):
    '''!
    read all records in a given path
    '''
    records = []
    (_, folders, frames) = next(os.walk(images_path))
    if "not_categorized" in folders:
        folders.remove("not_categorized")
    if "difficult_not_labeled" in folders:
        folders.remove("difficult_not_labeled")
    if len(folders) != 0:
        for subfolder in folders:
            sub_path = os.path.join(images_path, subfolder)
            records = records + record_reader(sub_path)
    else:
        labels_file = os.path.join(images_path, LABELS_FILE)
        labels = read_labels(labels_file)

        if LABELS_FILE in frames:
            frames.remove(LABELS_FILE)
        if "logo.jpg" in frames:
            frames.remove("logo.jpg")
        if "desktop.ini" in frames:
            frames.remove("desktop.ini")

        for frame in frames:
            full_frame_path = os.path.join(images_path, frame)
            records.append(Record(full_frame_path, labels))
    return records


class LabelClass(object):
    '''!
        Description of a label class
    '''
    def __init__(self, size, positions):
        # image path relative to dataset root
        self.size = size
        # array of 'Label' objects
        self.positions = positions

def get_unique_labels(records):
    '''!
    Extract all unique labels in a given record.
    '''
    all_labels = {}
    for rec in records:
        for label in rec.labels:
            if label.name in all_labels:
                if all_labels[label.name].size != label.pos[2:]:
                    print("{} - {}".format(rec.img_path, label.name))
                    print("{} vs {}".format(all_labels[label.name].size, label.pos[2:]))
                    assert False, "label size mismatch"
                else:
                    if label.pos[0:2] not in all_labels[label.name].positions:
                        all_labels[label.name].positions.append(label.pos[0:2])
            else:
                all_labels[label.name] = LabelClass(label.pos[2:], [label.pos[0:2]] )
    return all_labels

def remove_labels_corner(records, corner, corner_size=[140, 90], img_size=[720, 400]):
    '''!
    Remove labels which are not in the given corner
    '''
    if (True, True) == corner:
        x_limit, y_limit = corner_size
    elif (True, False) == corner:
        x_limit = img_size[0] - corner_size[0]
        y_limit = corner_size[1]
    elif (False, True) == corner:
        x_limit = corner_size[0]
        y_limit = img_size[1] - corner_size[1]
    elif (False, False) == corner:
        x_limit = img_size[0] - corner_size[0]
        y_limit = img_size[1] - corner_size[1]

    top, left = corner
    for idx, rec in enumerate(records):
        for label in rec.labels:
            pos = label.pos
            if (left and pos[0] + pos[2] > x_limit) or \
               (not left and pos[0] < x_limit) or \
               (top and pos[1] + pos[3] > y_limit) or \
               (not top and pos[1] < y_limit):
                rec.labels.remove(label)
        records[idx] = rec
    return records
