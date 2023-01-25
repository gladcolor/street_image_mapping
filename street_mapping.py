# Huan Ning
import glob
import os

from PIL import Image
import PIL
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
# from pano import GSV_pano

TARGET_IDS=[5, 7, 9, 10, 11, 16, 24, 30, 35, 40, 45]
# [0, 1, 2, 3, 4, 6, 8, 12, 14, 21, 22, 23, 36, 37, 38, 39, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53]
# Road surface classes:[5, 7, 9, 10, 11, 16, 24, 30, 35, 40, 45]
# gsv_pano = GSV_pano(panoId='vwC_g5HdcA33hnUDprQ7ag', saved_path=r'.\test_images')



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

    def is_edge(self, col, row, edge_threshold_pix=3):
        # edge_threshold = 3  # pixel
        is_touched = all_pair_list[int(idx / 2)][-1]
        if col < edge_threshold_pix or col > (self.igm_w - edge_threshold_pix - 1):
            is_touched = True

        if row < edge_threshold_pix or row > (self.igm_h - edge_threshold_pix - 1):
            is_touched = True

        if end_x < edge_threshold_pix or end_x > (self.igm_w - edge_threshold_pix - 1):
            is_touched = True

        if end_y < edge_threshold_pix or end_y > (self.igm_h - edge_threshold_pix - 1):
            is_touched = True

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
        self.detection_df['bearing_rad'] = math.radians(bearing_deg)
        self.detection_df['offset_x'] =  np.sin(self.detection_df['bearing_rad']) * self.detection_df['distance']
        self.detection_df['offset_y'] =  np.cos(self.detection_df['bearing_rad']) * self.detection_df['distance']


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

# def get_offset(distance, pano_yaw_deg, phi_deg):
#     bearing_rad = math.radians(pano_yaw_deg + phi_deg)
#     y_offset = distance * math.cos(bearing_rad)
#     x_offset = distance * math.sin(bearing_rad)
#     return x_offset, y_offset



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


class Image_landcover(object):
    def __init__(self, landcover_path, ):

        '''
        # How to measure road width?
        1. Generate binary image by target classes.
        2.
        '''

        self.invalid_edge_value = 255 # use 255 to indicate edge pixels

        self.landcover_path = landcover_path
        self.landcover_pil = Image.open(self.landcover_path)
        self.landcover_np = np.array(self.landcover_pil)
        # self.landcover_cv = cv2.cvtColor(self.landcover_np, cv2.COLOR_RGB2BGR)
        self.landcover_cv = cv2.imread(landcover_path)

        self.landcover_h, self.landcover_w = self.landcover_np.shape
        self.invalid_touch_np = None

        self.scaned_lines = None
        self.landcover_ext = landcover_path.split('.')[-1]

        ''' self.scaned_lines: a numpy array
        row: scan lines
        col0: row number
        col1: col number
        col2: length (pixel)
        '''
        # run_lengths: (col number, length), run_rows: row number

    def scan_width(self, pano_bearing_deg, target_ids,
                   morph_kernel_open=(5, 5),
                   morph_kernel_close=(11, 11),
                   interval_pix=5,
                   ):

        # preprocessing
        self.pano_bearing_deg = pano_bearing_deg
        helper.fill_image_edge(img_np=self.landcover_np, fill_value=self.invalid_edge_value)
        self.target_img_np = np.zeros(self.landcover_np.shape)

        for class_ in target_ids:
            self.target_img_np = np.logical_or(self.target_img_np, self.landcover_np == class_)
        self.smoothed_target = helper.img_smooth(self.target_img_np, open_kerel=(25, 25), close_kernel_close=(15, 15))



        self.rotated_target = helper.cv_img_rotate_bound(self.smoothed_target, -pano_bearing_deg)
        self.rotated_landcover_cv = helper.cv_img_rotate_bound(self.landcover_cv, -pano_bearing_deg)
        self.rotated_landcover_np = helper.cv_img_rotate_bound(self.landcover_np, -pano_bearing_deg)


        opened_color = cv2.merge((self.rotated_target, self.rotated_target, self.rotated_target))

        img_roatated_h = self.rotated_target.shape[0]
        line_y_list = range(interval_pix, img_roatated_h, interval_pix)
        to_RLE = self.rotated_target[line_y_list]
        run_lengths, run_rows = helper.rle_encoding(to_RLE, keep_nearest=False)
        # run_lengths, run_rows = helper.rle_encoding(to_RLE, keep_nearest=True)
        # run_lengths: (col number, length), run_rows: row number
        self.scaned_lines = None # np.ones((len(run_rows), 4)) * -1
        '''
        row: scan lines
        col0: row number
        col1: col number
        col2: length (pixel)
        col3: cover ratios
        '''

        lengths = run_lengths[1::2]  # get the lengths
        lengths = np.array(lengths)
        new_lengths = lengths.copy()

        new_run_cols = run_lengths[::2].copy()

        angle_deg = 0   # do not remember why use this variable, seems no useful. Maybe for rotation.
        angle_rad = math.radians(angle_deg)
        pen_lengths = lengths * math.cos(angle_rad)  #
        to_x = (pen_lengths * math.cos(angle_rad)).astype(int)
        to_y = (pen_lengths * math.sin(angle_rad)).astype(int)
        # process each row
        row_in_rotated_img = []
        for idx, col in enumerate(run_lengths):
            if idx % 2 == 0:
                idx2 = int(idx / 2)
                row = run_rows[idx2] * interval_pix + interval_pix - 1  # row in rotated img
                row = int(row)
                row_in_rotated_img.append(row)
                radius = 5

                length = run_lengths[idx + 1]
                new_run_cols[idx2] = col

                new_lengths[idx2] = length
                new_run_cols[idx2] = new_run_cols[idx2]

                # print("col, new_run_cols[idx2]:", col, new_run_cols[idx2])
                col = new_run_cols[idx2]
                col = int(col)
                to_x[idx2] = new_lengths[idx2]


        self.scaned_lines = np.array([row_in_rotated_img, new_run_cols, new_lengths]).T
        self.calculate_cover_ratio()

        self.scaned_lines_xy = self.compute_scaned_lines_xy()


        # find contour
        # raw_contours, hierarchy = cv2.findContours(img_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    def compute_scaned_lines_xy(self):
        tx = self.rotated_landcover_np.shape[0] / 2
        ty = self.rotated_landcover_np.shape[1] / 2

        end_points = np.zeros((len(self.scaned_lines) * 2, 2))

        # self.scaned_lines = row_in_rotated_img, new_run_cols, new_lengths
        end_points[:, 0] = np.append(self.scaned_lines[:, 1], self.scaned_lines[:, 1] + self.scaned_lines[:, 2])
        end_points[:, 1] = np.append(self.scaned_lines[:, 0], self.scaned_lines[:, 0])
        end_points_transed = helper.points_2D_translation(end_points, tx, ty)

        tx = self.target_img_np.shape[0] / 2
        ty = self.target_img_np.shape[1] / 2

        end_points_rotated = helper.points_2D_rotated(end_points_transed, -self.pano_bearing_deg)
        end_points_transed = helper.points_2D_translation(end_points_rotated, -tx, ty)

        # read worldfile
        worldfile_ext = self.landcover_path[-3] + self.landcover_path[-1] + 'w'
        worldfile_path = self.landcover_path[:-3] + worldfile_ext
        self.resolution, self.UL_x, self.UL_y = helper.read_worldfile(worldfile_path)

        x = end_points_transed[:, 0] * self.resolution + self.UL_x
        y = self.UL_y - end_points_transed[:, 1] * self.resolution

        scanline_cnt = len(self.scaned_lines)
        start_x = x[:scanline_cnt]
        start_y = y[:scanline_cnt]
        end_x = x[scanline_cnt:]
        end_y = y[scanline_cnt:]

        xy = np.array([start_x, start_y, end_x, end_y]).T

        return xy
    def show_img(self, img_type='img_rotated', scanline=True,
                 min_cover_ratio=0.90, min_length_pix=60, multiply_factor=100, line_thickness=1, radius=5):
        assert self.scaned_lines.shape[1] >= 4, "Need to compute the cover ratio in advance!"
        img = getattr(self, img_type) * multiply_factor
        if isinstance(img, (np.ndarray, np.generic)):
            if len(img.shape) == 2:
                img_to_show = cv2.merge((img, img, img))
            if len(img.shape) == 3:
                img_to_show = img #* 255
        if isinstance(img, PIL.TiffImagePlugin.TiffImageFile):
            img_to_show = img

        for idx, (start_row, start_col, length, cover_ratio, is_touch_invalid, is_touch_valid) in enumerate(self.scaned_lines[:, :6]):
            if (cover_ratio < min_cover_ratio) :  # and (not is_touch_valid)
                continue
            if length < min_length_pix:
                continue
            if is_touch_invalid:
                continue

            # if is_touch_valid:
            #     continue

            start_row = int(start_row)
            start_col = int(start_col)

            end_col = int(start_col + length)
            col = int((start_col + end_col) / 2)
            end_y = start_row
            # cover_ratio = helper.get_cover_ratio(col=col, row=row, mask_np=self.img_rotated,
            #                                      width=length, height=length)
            if scanline:
                cv2.line(img_to_show, (start_col, start_row), (end_col, end_y), (0, 0, 255), thickness=line_thickness)
                cv2.circle(img_to_show, (start_col, start_row), radius, (0, 255, 0), line_thickness)

        cv2.imshow(img_type, img_to_show)
        # waits for user to press any key
        # (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(0)

        # closing all open windows
        cv2.destroyAllWindows()

    def calculate_cover_ratio(self):

        if self.scaned_lines is None:
            print("Scanline is none. Please scan the land-cover image.")
            return
        cover_ratio_list = []
        for idx, (start_row, start_col, length) in enumerate(self.scaned_lines):
            row = start_row
            end_col = start_col + length
            col = int((start_col + end_col) / 2)
            cover_ratio = helper.get_cover_ratio(col=col, row=row, mask_np=self.rotated_target,
                                                 width=length, height=length)
            cover_ratio_list.append(cover_ratio)

        cover_ratio_list = np.array(cover_ratio_list).reshape((-1, 1))

        self.scaned_lines = np.append(self.scaned_lines, cover_ratio_list, axis=1)

    def set_invalid_touch(self, touch_ids):
        self.invalid_touch_np = np.zeros(self.rotated_landcover_np.shape)
        for class_ in touch_ids:
            self.invalid_touch_np = np.logical_or(self.invalid_touch_np, self.rotated_landcover_np == class_)

        self.invalid_touch_np = self.invalid_touch_np.astype(np.int8)

    def set_valid_touch(self, touch_ids):
        self.valid_touch_np = np.zeros(self.rotated_landcover_np.shape)
        for class_ in touch_ids:
            self.valid_touch_np = np.logical_or(self.valid_touch_np, self.rotated_landcover_np == class_)

        self.valid_touch_np = self.valid_touch_np.astype(np.int8)
    def scanlines_touch_validity(self, tolerance_pix=6):
        '''
        wether the scaneline ends tough the occluded or valid object.
        :param tolerance_pix:
        :return:
        '''
        if self.scaned_lines is None:
            print("Scanline is none. Please scan the land-cover image.")
            return
        is_touch_invalid_list = []
        is_touch_valid_list = []
        for idx, (start_row, start_col, length) in enumerate(self.scaned_lines[:, :3]):
            row = start_row
            end_col = start_col + length
            col = int((start_col + end_col) / 2)
            is_touch_invalid = helper.line_ends_touched(start_col, start_row, end_col, start_row, self.invalid_touch_np,
                                                  width=13, height=1, threshold=tolerance_pix, mode='or')

            # both ends need to touch the valid categories
            is_touch_valid = helper.line_ends_touched(start_col, start_row, end_col, start_row, self.valid_touch_np,
                                                        width=13, height=1, threshold=tolerance_pix, mode='and')

            is_touch_invalid_list.append(is_touch_invalid)
            is_touch_valid_list.append(is_touch_valid)

        is_touched_list = np.array(is_touch_invalid_list).reshape((-1, 1))
        self.scaned_lines = np.append(self.scaned_lines, is_touched_list, axis=1)

        is_touched_list = np.array(is_touch_valid_list).reshape((-1, 1))
        self.scaned_lines = np.append(self.scaned_lines, is_touched_list, axis=1)

    def remove_long_scaned_lines(self, max_length_pix=-1):  # not finished
        try:
            if max_length_pix == -1:
                max_length_pix = self.rotated_target.shape[1]
            # if (length > max_width_pix) and (idx2 > 0):  # if the scaned line is too long, use the previous length
                #     length = new_lengths[idx2 - 1]
                #     new_run_cols[idx2] = new_run_cols[idx2 - 1]
                #     print("long length!")
        except Exception as e:
            print("Error in remove_long_scaned_lines():", e)


    def save_scanline(self, save_dir='', file_name=''):  # , CRS="EPSG:4326"

        if (save_dir == '') and (file_name == '') and (self.landcover_path == ''):
            print("Error in Image_detection.save(): Need to set save_dir or file_name parameter.")

        if (file_name == '') and (save_dir != ''):
            os.makedirs(save_dir, exist_ok=True)
            basename = os.path.basename(self.landcover_path)
            file_name = os.path.join(save_dir, basename.replace(self.landcover_ext, 'csv'))

        else:
            if os.path.exists(self.landcover_path):
                file_name = self.landcover_path.replace(self.landcover_ext, 'csv')
                save_dir = os.path.dirname(self.landcover_path)


        self.scaned_lines_df = pd.DataFrame(self.scaned_lines_xy)
        self.scaned_lines_df.columns = ['start_x', 'start_y', 'end_x', 'end_y']
        self.scaned_lines_df['cover_ratio'] = self.scaned_lines[:, 3]

        self.scaned_lines_df['touch_invalid'] = self.scaned_lines[:, 4]
        self.scaned_lines_df['touch_valid'] = self.scaned_lines[:, 5]
        self.scaned_lines_df['length'] = self.scaned_lines[:, 2] * self.resolution
        self.scaned_lines_df['file_name'] = basename.replace('.' + self.landcover_ext, "")

        self.scaned_lines_df = self.scaned_lines_df.round(3)
        self.scaned_lines_df['touch_invalid'] = self.scaned_lines_df['touch_invalid'].astype(int)
        self.scaned_lines_df['touch_valid'] = self.scaned_lines_df['touch_valid'].astype(int)

        self.scaned_lines_df.to_csv(file_name, index=False)

        helper.measurements_to_shapefile(widths_files=[file_name], saved_path=save_dir)

        # col0: row number
        # col1: col number
        # col2: length (pixel)
        # col3: cover ratios