import os
import random

import torch
from torch.utils.data import Dataset

import pandas as pd
from PIL import Image

class BUSISegmentationDataset(Dataset):
    def __init__(self, dataset_dir, mode, transform=None, target_transform=None):
        super(BUSISegmentationDataset, self).__init__()
        self.image_folder = 'image'
        self.label_folder = 'mask'
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_transform = target_transform
        self.frame = pd.read_csv(os.path.join(dataset_dir, '{}_frame.csv'.format(mode)))


        # from sklearn.model_selection import train_test_split
        # self.frame = pd.read_csv(os.path.join(dataset_dir, 'metadata.csv'))
        # train, test = train_test_split(self.frame, test_size=0.25, shuffle=False, random_state=4321)
        # # train, val = train_test_split(train, test_size=0.11, shuffle=False, random_state=4321)
        #
        # train = train.drop(['Image_Id'], axis=1).reset_index().drop(['index'], axis=1)
        # test = test.drop(['Image_Id'], axis=1).reset_index().drop(['index'], axis=1)
        #
        # print(len(train))
        # print(len(test))
        #
        # print(train)
        #
        # # for idx, image_path in enumerate(list(train.image_path)):
        # #     train.loc[idx, 'new_mask_path'] = "{}_segmentation.png".format(image_path.split('.')[0])
        # # for idx, image_path in enumerate(list(val.image_path)):
        # #     val.loc[idx, 'new_mask_path'] = "{}_segmentation.png".format(image_path.split('.')[0])
        # # for idx, image_path in enumerate(list(test.image_path)):
        # #     test.loc[idx, 'new_mask_path'] = "{}_segmentation.png".format(image_path.split('.')[0])
        #
        # train.to_csv(os.path.join(dataset_dir, 'train_frame.csv'))
        # test.to_csv(os.path.join(dataset_dir, 'test_frame.csv'))
        #
        # sys.exit()

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        type = self.frame.image_path[idx].split()[0]
        image_path = os.path.join(self.dataset_dir, type, self.image_folder, self.frame.image_path[idx])
        label_path = os.path.join(self.dataset_dir, type, self.label_folder, self.frame.mask_path[idx])

        image = Image.open(image_path).convert('L')
        label = Image.open(label_path).convert('L')

        if self.transform is not None or self.target_transform is not None:
            seed = random.randint(0, 2 ** 32)
            
            if self.transform is not None:
                self._set_seed(seed)
                image = self.transform(image)
                
            if self.target_transform is not None:
                self._set_seed(seed)
                label = self.target_transform(label)

        label[label >= 0.5] = 1; label[label < 0.5] = 0

        return image, label

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)