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

import geopandas as gpd

import shapely
import cv2

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

def cv_img_rotate_bound(image, angle, flags=cv2.INTER_NEAREST):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), flags=flags)


def shape_add_XY():
    shp_file = r'E:\Research\street_image_mapping\Maryland_panoramas\jsons.shp'
    out_CRS = "EPSG:6487"
    # Transformer.from_crs(in_epsg, out_epsg)
    gdf = gpd.read_file(shp_file).to_crs(out_CRS)
    gdf['X'] = gdf['geometry'].centroid.x
    gdf['Y'] = gdf['geometry'].centroid.y
    gdf.to_file(shp_file)

    return


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


def img_smooth(img_cv, open_kerel=(5, 5), close_kernel_close=(11, 11)):
    g_close = cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel_close)
    g_open = cv2.getStructuringElement(cv2.MORPH_RECT, open_kerel)

    target_np = img_cv.astype(np.uint8)

    cv2_closed = cv2.morphologyEx(target_np, cv2.MORPH_CLOSE, g_close)  # fill small gaps
    cv2_opened = cv2.morphologyEx(cv2_closed, cv2.MORPH_OPEN, g_open)

    # cv2_opened = np.where(cv2_opened == 0, 0, 255).astype(np.uint8)

    return  cv2_opened



def keep_nearest_measurements(run_lengths, run_rows, center_col):
    new_run_cols = run_lengths[::2].copy()
    lengths = run_lengths[1::2]
    measure_dict = {}

    if len(run_rows) == 0:
        return  run_lengths, run_rows

    row_cnt = max(run_rows)

    # for each row in the image, use two matrix rows to store left/right measurements.
    left_right_measures = np.ones((row_cnt * 2 + 2, 5)) * -1  # columns: row, col, length, near_col
    left_right_measures[::2, 1] = -99999    # left measurement's col
    left_right_measures[1::2, 1] = 99999  # right measurement's col

    for idx, row in enumerate(run_rows):
        try:
            col = new_run_cols[idx]
            if col < center_col:  # in the left side
                old_left_col = left_right_measures[row * 2, 1]
                if col > old_left_col:
                    left_right_measures[row * 2, 1] = col
                    left_right_measures[row * 2, 0] = row
                    left_right_measures[row * 2, 2] = lengths[idx]
                    left_right_measures[row * 2, 3] = col + lengths[idx]
            if col > center_col:  # in the right side
                old_right_col = left_right_measures[row * 2 + 1, 1]
                if col < old_right_col:
                    left_right_measures[row * 2 + 1, 1] = col
                    left_right_measures[row * 2 + 1, 0] = row
                    left_right_measures[row * 2 + 1, 2] = lengths[idx]
                    left_right_measures[row * 2, 3] = col
        except Exception as e:
            # logging.error(str(e))
            print("Error in keep_nearest_measurements():", e)
            continue

    result = left_right_measures[left_right_measures[:, 2] > -1]

    return result[:, 1:3].reshape((-1,)), result[:, 0]

def rle_encoding(x, keep_nearest=False):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
    '''
    rows, cols = np.where(x == 1) # .T sets Fortran order down-then-right
    run_lengths = []
    run_rows = []
    prev = -2
    for idx, b in enumerate(cols):
        if (b != prev+1):  # in x-axis, skip to a non-adjacent column, start a new record
            run_lengths.extend((b, 0))     # record col number and length (start at 0 pixel)
            run_rows.extend((rows[idx],))  # record row number
        # else:  #(b < prev),  new line
        #     pass
        run_lengths[-1] += 1 # add a pixel to the length
        prev = b     # move to the next pixel's column

    if keep_nearest:
        center_col = x.shape[1] / 2
        run_lengths, run_rows = keep_nearest_measurements(run_lengths, run_rows, center_col)

    return run_lengths, run_rows   # run_lengths: (col number, length), run_rows: row number

if __name__ == '__main__':
    # convert_labelme_to_YOLOv5_txt()
    shape_add_XY()