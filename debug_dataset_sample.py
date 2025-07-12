import sys, torch
from torchvision import transforms
from dataset.BioMedicalDataset.BUSISegmentationDataset import BUSISegmentationDataset

dataset_dir='dataset/BioMedicalDataset/BUSI'
transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
mask_transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

ds=BUSISegmentationDataset(dataset_dir, mode='train', transform=transform, target_transform=mask_transform)
print('Dataset length', len(ds))
img, mask = ds[0]
print('img type', type(img), 'shape', img.shape)
print('mask type', type(mask), 'shape', mask.shape)
print('mask unique values', mask.unique()) 