# -*- coding:utf-8 -*-
import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os


class VOCDataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None, is_valid=False):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.root / "ImageSets/Main/test.txt"
        elif is_valid:
            image_sets_file = "/root/yolov3/data/cls62-val.txt"
        else :
            image_sets_file = "/root/yolov3/data/cls62-train.txt"
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        # if the labels file exists, read in the class names
        label_file_name = self.root / "labels.txt"
        
        # logging.info("Use custom labels.")

        self.class_names = ('car_front', 'car_etc', 'car_back', 'car_plate', 
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '가', '거', '경', '고',
            '구', '기', '나', '너', '노', '누', '다', '더', '도', '두', '라', '러', '로', '루',
            '리', '마', '머', '모', '무', '바', '배', '버', '보', '부', '사', '서', '소', '수',
            '시', '아', '어', '오', '우', '울', '임', '자', '장', '저', '조', '주', '청', '하',
            '허', '호')
            
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels = self._get_annotation(image_id)
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                tt = os.path.splitext(line.rstrip().split('/')[-1])[0]
                ids.append(tt)
        return ids

    def _get_annotation(self, image_id):
        annotation_file = f"/data/yper_data/converted_xml/{image_id}.xml"
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        boxes = []
        labels = []
        is_difficult = []

        main_body = []

        for object in root.findall('smr_object') :
            class_name = object.find('name').text
            if class_name in ['car_front', 'car_etc', 'car_back'] :
                for bbox in object.findall('bndbox') :
                    x1 = float(bbox.find('xmin').text)
                    y1 = float(bbox.find('ymin').text)
                    x2 = float(bbox.find('xmax').text)
                    y2 = float(bbox.find('ymax').text)

                    main_body = [x1, y1, x2, y2]
                    boxes.append(main_body)

                    labels.append(self.class_dict[class_name])
                    break
            else :
                continue
        for object in root.findall('smr_object') :
            class_name = object.find('name').text
            if class_name == 'car_plate' :
                lp_label = object.find('pose').text
                for bbox in object.findall('bndbox') :
                    x1 = float(bbox.find('xmin').text)
                    y1 = float(bbox.find('ymin').text)
                    x2 = float(bbox.find('xmax').text)
                    y2 = float(bbox.find('ymax').text)
                try :
                    if (x1 < main_body[0]) or (x2 > main_body[2]) or (y1 < main_body[1]) or (y2 > main_body[3]) :
                        continue
                    elif lp_label in list(self.class_names) :
                        boxes.append([x1, y1, x2, y2])
                        labels.append(self.class_dict[lp_label])
                except :
                    if lp_label in list(self.class_names) :
                        boxes.append([x1, y1, x2, y2])
                        labels.append(self.class_dict[lp_label])
                    else :
                        continue
        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))


            

    def _read_image(self, image_id):
        image_file = f"/data/yper_data/converted_img/resized_{image_id}.jpg"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image