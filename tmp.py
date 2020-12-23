from vision.datasets.voc_dataset import VOCDataset
from vision.utils.misc import store_labels
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.config import mobilenetv1_ssd_config
import os 
from vision.ssd.ssd import MatchPrior
from torch.utils.data import DataLoader
config = mobilenetv1_ssd_config
dataset_path = "/data/yper_data/converted_img"
train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

train_dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform)
label_file = "./labels.txt"
store_labels(label_file, train_dataset.class_names)
num_classes = len(train_dataset.class_names)
print(num_classes)
train_loader = DataLoader(train_dataset, 32,
                              num_workers=0,
                              shuffle=True)
for i, data in enumerate(train_loader):
    images, boxes, labels = data
    print(i)
    # print("boxes : ", boxes)
    # print("labels : ", labels)