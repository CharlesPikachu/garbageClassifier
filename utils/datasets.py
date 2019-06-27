'''
Function:
	load the dataset
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import os
import glob
import random
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


'''load data'''
class ImageFolder(Dataset):
	def __init__(self, data_dir, image_size, is_train=True, **kwargs):
		self.image_size = image_size
		self.image_paths = []
		self.image_labels = []
		self.classes = sorted(os.listdir(data_dir))
		for idx, cls_ in enumerate(self.classes):
			self.image_paths += glob.glob(os.path.join(data_dir, cls_, '*.*'))
			self.image_labels += [idx] * len(glob.glob(os.path.join(data_dir, cls_, '*.*')))
		self.indexes = list(range(len(self.image_paths)))
		if is_train:
			random.shuffle(self.indexes)
			self.transform = transforms.Compose([transforms.RandomResizedCrop(image_size),
												 transforms.RandomHorizontalFlip(),
												 transforms.ColorJitter(brightness=1, contrast=1, saturation=0.5, hue=0.5),
												 transforms.ToTensor(),
												 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
		else:
			self.transform = transforms.Compose([transforms.ToTensor(),
												 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	def __getitem__(self, index):
		image_path = self.image_paths[self.indexes[index]]
		image_label = self.image_labels[self.indexes[index]]
		image = Image.open(image_path).resize(self.image_size)
		image = self.transform(image)
		return image, image_label
	def __len__(self):
		return len(self.indexes)