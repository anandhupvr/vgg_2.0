import os
import random
import sys
import cv2
import numpy as np


class Data:

	def __init__(self, dataset):
		self.dataset_dir = dataset
		self.classes = os.listdir(dataset)
		self.train_count = ''
		self.build()

	def build(self):
		self.image_list = self._get_images()
		self.train_test_split()

	def _get_images(self):
		images_list = []
		for cat in self.classes:
			for im in os.listdir(os.path.join(self.dataset_dir, cat)):
				images_list.append(os.path.join(self.dataset_dir, cat, im))
		return images_list
	
	def train_test_split(self):
		random.shuffle(self.image_list)
		self.train_count = int(len(self.image_list)*.9)
		train_file = open('logs/train.txt','w')
		for img in self.image_list[:self.train_count]:
			train_file.write("%s\n"%img)
		test_file = open('logs/test.txt','w')
		for img in self.image_list[self.train_count:]:
			test_file.write("%s\n"%img)

	def _all_data(self):
		i = 1
		all_imgs = {}
		with open('logs/train.txt', 'r') as f:
			print ("Creating labels")
			for line in f:
				sys.stdout.write("\r"+"idx="+str(i))
				i += 1
				filename = line
				if filename not in all_imgs:
					all_imgs[filename] = {}
					all_imgs[filename]['path'] =filename
					cls_ = filename.split('/')[1]
					all_imgs[filename]['classes'] = self.classes.index(cls_)
			all_data = []
			for key in all_imgs:
				all_data.append(all_imgs[key])
		return all_data

	def get(self):
		all_data = self._all_data()
		batch_size = 4
		while True:
			j = 0
			for i in range(0, len(all_data), batch_size):
				img = []
				label_1hot = []
				imgs = all_data[i:i+batch_size]
				for im in imgs:
					label = np.zeros([len(self.classes)], dtype=np.float32)
					x_img = cv2.imread(im['path'].strip())
					x_img = cv2.resize(x_img, (224, 224))
					img.append(x_img)
					label[im['classes']] = 1
					label_1hot.append(label)
				j += 1
				yield np.array(img, dtype=np.float32), np.array(label_1hot)
	def numbers(self):
		return self.train_count