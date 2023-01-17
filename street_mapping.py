# Huan Ning
import glob
import os

from PIL import Image
import numpy as np
import helper
import requests
import math
import  pandas as pd
import cv2
# download py files.

URLs = [r"https://raw.githubusercontent.com/gladcolor/StreetView/master/gsv_pano/pano.py",
        r'https://raw.githubusercontent.com/gladcolor/StreetView/master/gsv_pano/utils.py',
        r'https://raw.githubusercontent.com/gladcolor/StreetView/master/gsv_pano/log_config.yaml']

# for URL in URLs:
#     response = requests.get(URL)
#     basename = os.path.basename(URL)
#     open(basename, "wb").write(response.content)

from pano import GSV_pano


# gsv_pano = GSV_pano(panoId='vwC_g5HdcA33hnUDprQ7ag', saved_path=r'.\test_images')

class Detection_info:
    def __init__(self):
        self.img_path = ''
        self.top_row = -1
        self.bottom_row = -1
        self.left_col = -1
        self.right_col = -1
        self.fov_h = -1
        self.fov_v = -1
        # self
class test1(object):
    def __init__(self, typein):
        print("Test 1 ok.", typein)

class Image_detection(object):

    def __init__(self, detection_file, img_ext='.jpg'):
        df = pd.read_csv(detection_file, header=None, sep=' ')  # , header=None

        if len(df.columns) == 5:
            columns = ['class', 'x_center', 'y_center', 'width_ratio', 'height_ratio']
        elif len(df.columns) == 6:
            columns = ['class', 'x_center', 'y_center', 'width_ratio', 'height_ratio', 'confidence']
        else:
            print("The number of columns should be 5 or 6!")

        df.columns = columns

        self.detection_df = df
        self.img_ext = img_ext
        # check the image
        img_name = detection_file.replace(".txt", img_ext)
        if os.path.exists(img_name):
            self.img_name = img_name
            img = Image.open(self.img_name)
            img_w, img_h = img.size
            img.close()

            self.set_img_w_h(img_w, img_h)

            self.compute_row_col()
        else:
            self.img_name = ''

    def set_img_w_h(self, img_w, img_h):
        self.img_w = img_w
        self.img_h = img_h
        self.detection_df['img_w'] = self.img_w
        self.detection_df['img_h'] = self.img_h
        self.compute_row_col()

    def compute_row_col(self):
        self.detection_df['top_row'] = self.img_h * (self.detection_df['y_center']
                                                     - self.detection_df['height_ratio'] / 2)
        self.detection_df['bottom_row'] = self.img_h * (self.detection_df['y_center']
                                                        + self.detection_df['height_ratio'] / 2)

        self.detection_df['left_col'] = self.img_w * (self.detection_df['x_center']
                                                      - self.detection_df['width_ratio'] / 2)
        self.detection_df['right_col'] = self.img_w * (self.detection_df['x_center']
                                                       + self.detection_df['width_ratio'] / 2)

        self.detection_df['c_col'] = (self.detection_df['right_col'] + self.detection_df['left_col']) / 2


    def compute_distance(self, bar_length_dict, fov_h_deg): # bar_length_dict as {'class_index': bar_length}
        class_ids = list(bar_length_dict.keys())
        self.detection_df[self.detection_df['class'].isin(class_ids)]
        self.bar_length_dict = bar_length_dict
        self.fov_h_deg = fov_h_deg
        self.detection_df['fov_h_deg'] = self.fov_h_deg
        self.detection_df['bar_length'] = self.detection_df[['class']].replace(bar_length_dict)['class'].to_list()

        self.compute_phi_theta()

        distances = self.detection_df['bar_length'] * np.cos(self.detection_df['top_theta']) * np.cos(
            self.detection_df['bottom_theta']) / np.sin(
            self.detection_df['top_theta'] - self.detection_df['bottom_theta'])

        self.detection_df['distance'] = np.array(distances)

        self.detection_df['distance_z'] = self.detection_df['distance'] * np.tan(self.detection_df['bottom_theta'])

        # fov_h_rad = math.radians(fov_h_deg)
        # col = col - width / 2  # move the origin to center
        # row = height / 2 - row
        # fov_v = atan((height * tan((fov_h_rad / 2)) / width)) * 2
        # r = (width / 2) / tan(fov_h_rad / 2)
        # s = sqrt(col ** 2 + r ** 2)
        # theta = atan(row / s)
        # phi = atan(col / r)

    def compute_phi_theta(self):
        fov_h_rad = math.radians(self.fov_h_deg)

        # move the origin to center
        cols = np.array(self.detection_df['c_col'] - self.detection_df['img_w'] / 2)
        top_rows = np.array(self.detection_df['img_h'] / 2 - self.detection_df['top_row'])
        bottom_rows = np.array(self.detection_df['img_h'] / 2 - self.detection_df['bottom_row'])
        r = np.array((self.detection_df['img_w'] / 2) / math.tan(fov_h_rad / 2))
        s = np.sqrt(cols ** 2 + r ** 2)
        self.detection_df['phi']   = np.arctan(cols / r)    # radians
        self.detection_df['top_theta'] = np.arctan(top_rows / r) # radians
        self.detection_df['bottom_theta'] = np.arctan(bottom_rows / r) # radians

    def save(self, save_dir='', file_name=''):

        if (save_dir == '') and (file_name == '') and (self.img_name == ''):
            print("Error in Image_detection.save(): Need to set save_dir or file_name parameter.")

        if (file_name == '') and (save_dir != ''):
            os.makedirs(save_dir, exist_ok=True)
            basename = os.path.basename(file_name)
            file_name = os.path.join(save_dir, basename.replace(self.img_ext, '.csv'))

        if os.path.exists(self.img_name):
            file_name = self.img_name.replace(self.img_ext, '.csv')

        self.detection_df.to_csv(file_name, index=False)

    def compute_offset(self, bearing_deg):  # , pano_yaw_deg

        if 'distance' not in self.detection_df.columns:
            self.compute_distance()

        # self.detection_df['bearing'] = self.detection_df['phi'] + math.radians(pano_yaw_deg)
        self.detection_df['bearing'] = math.radians(bearing_deg)
        self.detection_df['offset_x'] =  np.sin(self.detection_df['bearing']) * self.detection_df['distance']
        self.detection_df['offset_y'] =  np.cos(self.detection_df['bearing']) * self.detection_df['distance']


def tacheometry_distance(bar_length, top_row, bottom_row, c_col, fov_h_deg, img_h, img_w):
    '''

    :param bar_length: unit:meter
    :param top_row:
    :param bottom_row:
    :param c_col:
    :param fov_h: unit: degree
    :param img_h:
    :param img_w:
    :return:
    '''
    # distance = 0

    top_phi, top_theta       = helper.castesian_to_shperical(c_col, top_row, fov_h_deg, img_h, img_w)
    bottom_phi, bottom_theta = helper.castesian_to_shperical(c_col, bottom_row, fov_h_deg, img_h, img_w)
    distance = bar_length * math.cos(top_theta) * math.cos(bottom_theta) / math.sin(top_theta - bottom_theta)

    sink_distance = distance * math.tan(bottom_theta)

    # y_offset = distance * math.cos(bottom_phi)
    # x_offset = distance * math.sin(bottom_phi)

    # castesian_to_shperical(col, row, fov_h_deg, height, width)

    return distance #, x_offset, y_offset, sink_distance

def get_offset(distance, pano_yaw_deg, phi_deg):
    bearing_rad = math.radians(pano_yaw_deg + phi_deg)
    y_offset = distance * math.cos(bearing_rad)
    x_offset = distance * math.sin(bearing_rad)
    return x_offset, y_offset



def _get_object_bboxes(txt_file_path, label_idx):
    print(txt_file_path)
    f = open(txt_file_path, 'r')
    lines = f.readlines()
    f.close()

    lines = [line.replace("\n", "") for line in lines]
    objects_fileds = [line.split(' ') for line in lines]
    # print(objects_fileds)
    img_name = txt_file_path.replace(".txt", '.jpg')
    img = Image.open(img_name)
    img_w, img_h = img.size
    img.close()
    img_basename = os.path.basename(img_name)[:-4]

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

            return top_row, right_col, bottom_row, left_col, img_w, img_h

        except Exception as e:
            print("Error in _get_object_bbox():", txt_file_path, e)

    return

def yolo5_bbox_to_distance(bbox_dir, bar_length, fov_h_deg, label_idx):
    '''

    :param bbox_dir:
    :param bar_length:
    :param fov_h: degrees
    :param label_idx:
    :return:
    '''
    txt_files = glob.glob(os.path.join(bbox_dir, '*.txt'))
    print("Found txt file count:", len(txt_files))

    for idx, txt_file in enumerate(txt_files[:]):
        top_row, right_col, bottom_row, left_col, img_w, img_h = _get_object_bboxes(txt_file, label_idx)
        c_col = (right_col + left_col)/2
        # print("top_row, right_col, bottom_row, left_col:", top_row, right_col, bottom_row, left_col)
        distance = tacheometry_distance(bar_length, top_row, bottom_row, c_col, fov_h_deg, img_h, img_w)  # , x_offset, y_offset, sink_distance
        print("distancee:", distance)

    return distance

class Bbox_mapping:

    def __int__(self, bar_length, top_row, right_col, bottom_row, left_col, fov_h_deg, img_h, img_w):
        self.bar_length = bar_length
        self.top_row = top_row
        self.bottom_row = bottom_row
        self.right_col = right_col
        self.left_col = left_col
        self.c_col = (right_col + left_col) / 2
        self.fov_h_deg = fov_h_deg
        self.img_h = img_h
        self.img_w = img_w

        self.top_phi, self.top_theta = helper.castesian_to_shperical(self.c_col, self.top_row, self.fov_h_deg, self.img_h, self.img_w)
        self.bottom_phi, self.bottom_theta = helper.castesian_to_shperical(self.c_col, self.bottom_row, self.fov_h_deg, self.img_h, self.img_w)
        self.distance = self.bar_length * math.cos(self.top_theta) * math.cos(self.bottom_theta) / math.sin(self.top_theta - self.bottom_theta)

        self.sink_distance = self.distance * math.tan(self.bottom_theta)

    def get_offset(self, pano_yaw_deg):
        self.bearing = (pano_yaw_deg + self.top_phi)
        y_offset = self.distance * math.cos(self.bearing)
        x_offset = self.distance * math.sin(self.bearing)

        xyz_offset = (x_offset, y_offset, self.sink_distance)

        return xyz_offset
    # def distance(self):
    #     '''
    #
    #     :param bar_length: unit:meter
    #     :param top_row:
    #     :param bottom_row:
    #     :param c_col:
    #     :param fov_h: unit: degree
    #     :param img_h:
    #     :param img_w:
    #     :return:
    #     '''
    #     # distance = 0
    #
    #
    #
    #     # castesian_to_shperical(col, row, fov_h_deg, height, width)
    #
    #     return self.distance  # , x_offset, y_offset, sink_distance

    # @staticmethod
    # def map_width(DOM_img, heading_deg: float):
    #
    #     return

class Image_segmentation(object):
    def __init__(self, seg_path, ):


        self.seg_path = seg_path

class Image_landcover(object):
    def __init__(self, landcover_path, ):

        '''
        # How to measure road width?
        1. Generate binary image by target classes.
        2.
        '''

        self.landcover_path = landcover_path
        self.landcover_pil = Image.open(self.landcover_path)
        self.landcover_np = np.array(self.landcover_pil)
        self.landcover_h, self.landcover_w = self.landcover_np.shape
        self.scaned_lines = None


    def scan_width(self, pano_bearing_deg, target_ids,
                   morph_kernel_open=(5, 5),
                   morph_kernel_close=(11, 11),
                   interval_pix=10,
                   ):

        # preprocessing
        self.target_img_np = np.zeros(self.landcover_np.shape)
        for class_ in target_ids:
            self.target_img_np = np.logical_or(self.target_img_np, self.landcover_np == class_)
        self.img_smoothed = helper.img_smooth(self.target_img_np, open_kerel=(25, 25), close_kernel_close=(15, 15))
        self.img_rotated = helper.cv_img_rotate_bound(self.img_smoothed, -pano_bearing_deg)

        opened_color = cv2.merge((self.img_rotated, self.img_rotated, self.img_rotated))

        img_roatated_h = self.img_rotated.shape[0]
        line_y_list = range(interval_pix, img_roatated_h, interval_pix)
        to_RLE = self.img_rotated[line_y_list]
        # run_lengths, run_rows = helper.rle_encoding(to_RLE, keep_nearest=False)
        run_lengths, run_rows = helper.rle_encoding(to_RLE, keep_nearest=True)
        # run_lengths: (col number, length), run_rows: row number

        lengths = run_lengths[1::2]  # get the lengths
        lengths = np.array(lengths)
        new_lengths = lengths.copy()

        new_run_cols = run_lengths[::2].copy()

        max_width_pix = self.img_rotated.shape[1]

        is_show_img = True

        angle_deg = 0   # do not remember why use this variable
        line_thickness = 1
        angle_rad = math.radians(angle_deg)
        pen_lengths = lengths * math.cos(angle_rad)
        to_x = (pen_lengths * math.cos(angle_rad)).astype(int)
        to_y = (pen_lengths * math.sin(angle_rad)).astype(int)
        # process each row
        for idx, col in enumerate(run_lengths):
            if idx % 2 == 0:
                idx2 = int(idx / 2)
                row = run_rows[idx2] * interval_pix + interval_pix - 1
                row = int(row)
                radius = 5
                # print(row, col)
                length = run_lengths[idx + 1]
                new_run_cols[idx2] = col
                # if (length > max_width_pix) and (idx2 > 0):  # if the scaned line is too long, use the previous length
                #     length = new_lengths[idx2 - 1]
                #     new_run_cols[idx2] = new_run_cols[idx2 - 1]
                #     print("long length!")
                    # print("length, new_lengths[idx2], max_width_pix:", length, new_lengths[idx2], max_width_pix)

                new_lengths[idx2] = length
                new_run_cols[idx2] = new_run_cols[idx2]

                # print("col, new_run_cols[idx2]:", col, new_run_cols[idx2])
                col = new_run_cols[idx2]
                col = int(col)
                to_x[idx2] = new_lengths[idx2]
                # to_y[idx2] = row

                end_x = col + to_x[idx2]
                end_x = int(end_x)
                end_y = row + to_y[idx2]
                end_y = int(end_y)

                if is_show_img:
                    cv2.line(opened_color, (col, row), (end_x, end_y), (0, 0, 255), thickness=line_thickness)
                    cv2.circle(opened_color, (col, row), radius, (0, 255, 0), line_thickness)

        self.scaned_lines = (new_run_cols, new_lengths, new_run_cols)

        if is_show_img:
            cv2.imshow("cv2_opened", opened_color)
            # waits for user to press any key
            # (this is necessary to avoid Python kernel form crashing)
            cv2.waitKey(0)

            # closing all open windows
            cv2.destroyAllWindows()
        # find contour
        # raw_contours, hierarchy = cv2.findContours(img_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)



    def remove_long_scaned_lines(self, max_length_pix=-1):
        try:
            if max_length_pix == -1:
                max_length_pix = self.img_rotated.shape[1]
            # if (length > max_width_pix) and (idx2 > 0):  # if the scaned line is too long, use the previous length
                #     length = new_lengths[idx2 - 1]
                #     new_run_cols[idx2] = new_run_cols[idx2 - 1]
                #     print("long length!")
        except Exception as e:
            print("Error in remove_long_scaned_lines():", e)


