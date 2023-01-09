from pyproj import Transformer, Proj, transform
import matplotlib.pylab as plt
import math
import pandas as pd
import numpy as np
import os
import glob
import glob
import os
import math
from PIL import Image
from tqdm import tqdm
import pandas as pd
import math
from math import *
from scipy.stats import norm
from sklearn.metrics import r2_score
import json


import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter

def list_to_file(a_list, file_name):
    with open(file_name, mode='w', encoding='utf-8') as myfile:
        myfile.write('\n'.join(a_list))
        myfile.write('\n')

def yolov5_detection_file_to_bbox(file_name, sep=' ', img_ext='.jpg'):
    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()

    img_name = file_name.replace(".txt", img_ext)
    img = Image.open(img_name)
    img_w, img_h = img.size
    img.close()
    # img_basename = os.path.basename(img_name)[:-4]

    lines = [line.replace("\n", "") for line in lines]
    objects_fileds = [line.split(sep) for line in lines]
    detection_list = []
    obj_id = -1
    conf = -1
    label_id = -1
    for object in objects_fileds:
        try:
            if len(object) == 6:
                obj_id = int(object[0])
                cxywh = object[1:]
                conf, x, y, w, h = [float(n) for n in cxywh]
            else:
                label_id, x, y, w, h = [float(n) for n in object]

            top_row = img_h * (y - h / 2)
            bottom_row = img_h * (y + h / 2)
            left_col = img_w * (x - w / 2)
            right_col = img_w * (x + w / 2)

            bbox = (obj_id, conf, label_id, top_row, right_col, bottom_row, left_col, img_w, img_h)
            detection_list.append(bbox)

            return detection_list

        except Exception as e:
            print("Error in _get_object_bbox():", file_name, e)

def castesian_to_shperical(col, row, fov_h_deg, height, width):  # yaw: set the heading, pitch
    """
    Convert the row, col to the  spherical coordinates
    :param row, cols:
    :param fov_h:
    :param height: perspective image height
    :param width:  perspective image width
    :return: direction angles: phi (horizontal), theta (vertical)
    """
    fov_h_rad = math.radians(fov_h_deg)
    col = col - width / 2  # move the origin to center
    row = height / 2 - row
    fov_v = atan((height * tan((fov_h_rad / 2)) / width)) * 2
    r = (width / 2) / tan(fov_h_rad / 2)
    s = sqrt(col ** 2 + r ** 2)
    theta = atan(row / s)
    phi = atan(col / r)

    return phi, theta




def convert_labelme_to_YOLOv5_txt():
    json_path = r'E:\Research\street_image_mapping\Maryland_panoramas\training_data2'
    # saved_path = r'E:\Research\street_image_mapping\Maryland_panoramas\training_data_yolo_lables'
    saved_path = json_path
    print(json_path)

    extension = 'json'
    files = glob.glob(os.path.join(json_path, "*." + extension))
    # files = natsorted(files)
    files = sorted(files)
    print(len(files))

    train_lines = []

    for jfile in files[:]:
        object_lines = []

        #     train_fields.append(jfile)
        #     print(jfile)
        f = open(jfile, 'r')
        jdata = json.load(f)
        f.close()

        basename = os.path.basename(jfile)

        shapes = jdata["shapes"]

        # label_list = ["door", "house", "step", 'garage']
        label_list = ['stop']

        img_w = jdata["imageWidth"]
        img_h = jdata["imageHeight"]

        for idx, shape in enumerate(shapes[:]):

            label = shape["label"]
            points = shape["points"]
            points = np.array(points)

            object_line = []

            train_fields = []

            try:
                label_idx = label_list.index(label)
                min_x = min(points[:, 0])
                max_x = max(points[:, 0])
                min_y = min(points[:, 1])
                max_y = max(points[:, 1])

                # print(max_x, max_y, min_x, min_y)

                center_x = (min_x + max_x) / 2
                center_x = center_x / img_w

                center_y = (min_y + max_y) / 2
                center_y = center_y / img_h

                h = (max_y - min_y) / img_h
                w = (max_x - min_x) / img_w

                object_line = f"{label_idx} {center_x:.6f} {center_y:.6f} {w:.6f} {h:.6f}"


                object_lines.append(object_line)


            except Exception as e:
                print("Error: ", e)
                continue

        # print(len(object_line))
        if len(object_line) > 0:
            new_name = os.path.join(saved_path, basename.replace(".json", '.txt'))
            list_to_file(object_lines, new_name)

            print(jfile)

        else:
            print("No object.")


if __name__ == '__main__':
    convert_labelme_to_YOLOv5_txt()