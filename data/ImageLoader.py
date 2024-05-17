import cv2
import numpy as np

class ImageLoader():
    def __init__(self, labels, imagePaths, image_transform):
        self.labels = labels
        self.imagePaths = imagePaths
        self.image_transform = image_transform

    def read_image(self, image_path):
        return self.image_transform(image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))["image"].astype(np.int32)

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, index):
        index = index % len(self)
        return self.read_image(self.imagePaths[index]), self.labels[index]

    def iter(self):
        for i in range(len(self)):
            yield self[i]

    def margin_format(self, image, label):
        return {'image': image, 'label': label}, label