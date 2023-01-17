import glob
import json
import os
import random

import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

import helper
from pano import  GSV_pano
import street_mapping as sm
import geopandas as gpd


def get_DOM():
    # get a DOM
    # pano1 = GSV_pano(panoId='TSU3yYhn7Lmx5BLQnSS8Jw', saved_path='.', crs_local='EPSG:6569')  # EPSG:6569: South Carolina
    # pano1 = GSV_pano(panoId='vwC_g5HdcA33hnUDprQ7ag', saved_path='.', crs_local='EPSG:6569')
    # pano1 = GSV_pano(panoId='-2YL71_3f551jKPxMSN4Gg', saved_path='.', crs_local='EPSG:6487')   # EPSG:6487 Maryland
    # pano1 = GSV_pano(panoId='1S2IJdqU_ZDwe-kF5ux-wg', saved_path='.', crs_local='EPSG:6487')   # EPSG:6487 Maryland
    pano1 = GSV_pano(panoId='Drgf5QkdTdhxZq7axLxNMw', saved_path=r'.\test_images', crs_local='EPSG:6487')   # EPSG:6487 Maryland
    pano1.get_DOM()

def compute_coordinate():

    pass

def test_Image_detection():
    detection_dir = r'E:\Research\street_image_mapping\yolov5\runs\detect\exp'
    detection_files = glob.glob(os.path.join(detection_dir, "*.txt"))
    print("Found file count: ", len(detection_files))

    for detection_file in tqdm(detection_files[:]):

        if 'Copy' in detection_file:
            continue

        # detection_file = r'./test_images/_gCN6R-X-3U6YqKZcuo1dA_0.0_0.0_0_133.41_F20.txt'  # error: 17.2 - 15.5 = 1.8 m

        img_detection = sm.Image_detection(detection_file=detection_file)

        img_detection.compute_distance(bar_length_dict={0:0.9}, fov_h_deg=20)
        basename = os.path.basename(detection_file)
        panoId = basename[:22]
        bearing_deg = basename.replace("fov=20.txt", '').split('_')[-1]
        bearing_deg = float(bearing_deg)
        # img_detection.compute_offset(bearing_deg=133.40961)
        img_detection.compute_offset(bearing_deg=bearing_deg)
        img_detection.detection_df['panoId'] = panoId
        img_detection.save()
        # ground_true = 17.2  # meter
        # error = abs(ground_true - img_detection.detection_df.iloc[0]['distance'])
        # print("Distance, ground_true, error:", img_detection.detection_df.iloc[0]['distance'], ground_true, error)
        # print(img_detection.detection_df)
# sm.yolo5_bbox_to_distance(bbox_dir=r'E:\Research\street_image_mapping\Maryland_panoramas\training_data', label_idx=0)
def get_all_DOMs():

    jsons = glob.glob(r'E:\Research\street_image_mapping\Maryland_panoramas\jsons\*.json')
    save_path = r'E:\Research\street_image_mapping\Maryland_panoramas\pano_DOM'
    cut = 5
    interval = 20001
    start_idx = interval * cut
    end_idx = min(len(jsons), start_idx + interval)
    print("start_idx, end_idx:", start_idx, end_idx)
    for json_file in tqdm(jsons[start_idx:end_idx]):  # 4000
        # print(json_file)
        try:
            pano1 = GSV_pano(json_file=json_file, saved_path=save_path, crs_local='EPSG:6487')   # EPSG:6487 Maryland
            pano1.get_DOM()
        except Exception as e:
            print("Error in get_all_DOMs():", json_file, e)
            continue

def sampling_training_data():
    all_img_dir = r'E:\Research\street_image_mapping\Maryland_panoramas\training_data2'
    all_img = glob.glob(os.path.join(all_img_dir, "*.txt"))
    all_img = [img.replace(".txt", ".jpg") for img in all_img]

    X_train, X_test, _, _ = train_test_split(all_img, all_img, test_size=0.2, random_state=88)
    print("Total sample, train sample, test sample:", len(all_img), len(X_train), len(X_test))
    helper.list_to_file(X_train, r'E:\Research\street_image_mapping\Maryland_panoramas\train.txt')
    helper.list_to_file(X_test, r'E:\Research\street_image_mapping\Maryland_panoramas\test.txt')

def add_offset_to_pano():
    detection_dir = r'E:\Research\street_image_mapping\yolov5\runs\detect\exp'
    detection_files = glob.glob(os.path.join(detection_dir, "*.csv"))
    print("Found file count: ", len(detection_files))

    shp_file = r'E:\Research\street_image_mapping\Maryland_panoramas\jsons.shp'
    gdf = gpd.read_file(shp_file)

    df_list = []
    for detection_file in tqdm(detection_files[:]):

        if 'Copy' in detection_file:
            continue

        # detection_file = r'./test_images/_gCN6R-X-3U6YqKZcuo1dA_0.0_0.0_0_133.41_F20.txt'  # error: 17.2 - 15.5 = 1.8 m

        # img_detection = sm.Image_detection(detection_file=detection_file)
        detection_df = pd.read_csv(detection_file)
        df_list.append(detection_df)
    print("Started to merge...")
    df = pd.concat(df_list)
    df.to_csv(r'E:\Research\street_image_mapping\detected_stop_sign.csv', index=False)
    # df = pd.read_csv(r'E:\Research\street_image_mapping\detected_stop_sign.csv')
    df = df.merge(gdf, left_on='panoId', right_on='panoId')
    df['sign_x'] = df['X'] + df['offset_x']
    df['sign_y'] = df['Y'] + df['offset_y']
    df['sign_z'] = df['elevatio_1'] + df['distance_z']

    df.to_csv(r'E:\Research\street_image_mapping\measured_stop_sign.csv', index=False)

    # geometry = gpd.points_from_xy(df['sign_x'], df['sign_y'], df['sign_z'])
    gdf2 = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['sign_x'], df['sign_y']))
    gdf2.to_file(r'E:\Research\street_image_mapping\measured_stop_sign.shp')
    print("Done.")

if __name__ == "__main__":
    # get_DOM()
    # get_all_DOMs()
    # sampling_training_data()
    # test_Image_detection()
    add_offset_to_pano()