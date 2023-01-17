import math
import unittest
import json

import helper
import street_mapping as sm
# import rasterio

class Street_mapping_test(unittest.TestCase):
    # def test_load_data(self):
    #     sm1 = sm.Street_mapping().map_width(DOM_img=0, heading_deg=0)
    #     json_file = r'./test_images/pzrQ0m1V_feSktSRq14H4g.json'
    #     true_heading_deg = 144.2741
    #     pano_json = json.load(open(json_file))
    #     heading_deg = pano_json['Projection']['pano_yaw_deg']
    #     heading_deg = round(heading_deg, 4)
    #     self.assertEqual(heading_deg, true_heading_deg)  # add assertion here

    def test_tacheometry_distance(self):

        # FHWA Regulatory Signs: https://mutcd.fhwa.dot.gov/htm/2003r1/part2/part2b1.htm
        # Stop sign: Convertional Road: 750*750 (30*30), Expressway:900*900，　 Minimum: 600*600

        # TSU3yYhn7Lmx5BLQnSS8Jw_yaw=155_fov=30.jpg
        distance = sm.tacheometry_distance(bar_length=0.9, top_row=250, bottom_row=377, c_col=310,
                                           fov_h_deg=30, img_h=768, img_w=1024)
        # print("distance: ", distance)
        distance = sm.tacheometry_distance(bar_length=0.9, top_row=376, bottom_row=485, c_col=468,
                                           fov_h_deg=30, img_h=768, img_w=1024)
        # print("distance: ", distance)  # 12.62 in DOM
        self.assertEqual(round(distance, 2), 15.78)

    # def test_yolo5_bbox_to_distance(self):
    #     file_cnt = sm.yolo5_bbox_to_distance(bbox_dir=r'E:\Research\street_image_mapping\Maryland_panoramas\training_data',
    #                                          bar_length=0.9, fov_h_deg=20, label_idx=0)
    #     # print("Txt file count:", file_cnt)
    #     # self.assertEqual(file_cnt, 233)
    #
    # def test_Bbox_mapping_get_offset(self):
    #     bbox_mapping1 = sm.Bbox_mapping(bar_length=0.9, top_row, right_col, bottom_row, left_col, fov_h_deg, img_h, img_w)
    #
    # def test_helper_yolov5_detection_file_to_bbox(self):
    #     detection_file = r'./test_images/Drgf5QkdTdhxZq7axLxNMw_0.0_0.0_0_134.00_F10.txt'
    #     detection_bboxes = helper.yolov5_detection_file_to_bbox(detection_file)
    #     print(detection_file, detection_bboxes)

    def test_Image_detection(self):
        detection_file = r'./test_images/_gCN6R-X-3U6YqKZcuo1dA_0.0_0.0_0_133.41_F20.txt'  # error: 17.2 - 15.5 = 1.8 m

        img_detection = sm.Image_detection(detection_file=detection_file)

        img_detection.compute_distance(bar_length_dict={0:0.9}, fov_h_deg=20)
        img_detection.compute_offset(pano_yaw_deg=133.40961)
        img_detection.save()
        ground_true = 17.2  # meter
        error = abs(ground_true - img_detection.detection_df.iloc[0]['distance'])
        # print(img_detection.detection_df)


        self.assertTrue(error < 2)


    def test_Image_segmentation(self):
        segmentation_file = r'./test_images/_gCN6R-X-3U6YqKZcuo1dA_0.0_0.0_0_133.41_F20.txt'

        img_detection = sm.Image_detection(detection_file=detection_file)

        img_detection.compute_distance(bar_length_dict={0:0.9}, fov_h_deg=20)
        img_detection.compute_offset(pano_yaw_deg=133.40961)
        img_detection.save()
        ground_true = 17.2  # meter
        error = abs(ground_true - img_detection.detection_df.iloc[0]['distance'])
        # print(img_detection.detection_df)


        self.assertTrue(error < 2)

    def test_Image_landcover(self):
        seg_file = r'./test_images/S_ZmoNLdzo0FApj_jGiFJg_DOM_0.05.tif'

        img_landcover = sm.Image_landcover(landcover_path=seg_file)

        # helper.img_smooth(img_cv=img_landcover.landcover_np)

        # plt.imshow(img_landcover.landcover_pil)
        img_landcover.scan_width(pano_bearing_deg=225, target_ids=[10, 16, 24, 30, 35])

        self.assertEqual(img_landcover.landcover_h, 800)
        self.assertEqual(img_landcover.landcover_w, 800)

if __name__ == '__main__':
    unittest.main()

        # self.assertEqual(round(distance, 2), 13.62)

        # vwC_g5HdcA33hnUDprQ7ag_yaw=90_fov=30.