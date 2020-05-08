# fit a mask rcnn on the kangaroo dataset
from os import listdir
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import pandas as pd
img_rows, img_cols = 1200, 675
Number_Images = len(listdir(downloads/vessels))
Fraction_Training_set=0.9
 
# class that defines and loads the dataset
class Vessel_Human_Dataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "Dangerous")
		self.add_class("dataset", 0, "Safe")
		# define data locations
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/csv/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			count +=1
			# we are building the train set, 90% of data
			if is_train and  count <= len(listdir(images_dir))*Fraction_Training_set:
				continue
			# we are building the test/val set, 10% of data
			if not is_train and count > len(listdir(images_dir))*Fraction_Training_set:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + '.csv'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
 
	# extract bounding boxes from an annotation file
	def extract_boxes(filename):
	# load and parse the file
	train = pd.read_csv(filename)
	boxes = list()
	dangerous_list = list()

	for box in range(len(train[bbox_id])):
		xmin= int(train['x_min'][box])
		ymin= int(train['y_min'][box])
		xmax= int(train['x_max'][box])
		ymax= int(train['y_max'][box])
		dangerous = int(train['Dangerous'][box])
		coors= [xmin,ymin, xmax, ymax]
		dangerous_list.append(dangerous)
		boxes.append(coors)
	return boxes,dangerous_list

	# load the masks for an image
	def load_mask(self, image_id, path):
		
		# load CSV
		h = img_rows
		w = img_cols
		boxes, danger_list = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1

			if danger_list[i] = 1:
				class_ids.append(self.class_names.index('Dangerous'))
			else:
				class_ids.append(self.class_names.index('Safe'))

		return masks, asarray(class_ids, dtype='int32')
 
 
	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']
 
# define a configuration for the model
class VesselConfig(Config):
	# define the name of the configuration
	NAME = "Vessel_cfg"
	# number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 1
	# number of training steps per epoch
	STEPS_PER_EPOCH = len(Number_Images*Fraction_Training_set)
 
# prepare train set
train_set = Vessel_Human_Dataset()
train_set.load_dataset('Vessel2', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# prepare test/val set
test_set = Vessel_Human_Dataset()
test_set.load_dataset('Vessel2', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# prepare config
config = VesselConfig()
config.display()
# define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')