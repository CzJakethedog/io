# import lib
from keras.utils import np_utils
import numpy as np
import h5py

class HDF5DatasetGenerator:
	def __init__(self, dbPath, batchSize, preprocessors=None, aug=None, binarize=True, classes=2):
		self.batchSize = batchSize
		self.preprocessors = preprocessors
		self.aug = aug
		self.binarize = binarize
		self.classes = classes
		# read database and determine the total number of entries in the database
		self.db = h5py.File(dbPath)
		self.numImages = self.db["labels"].shape[0]

	def generator(self, passes=np.inf):
		# initialize the epoch count
		epochs = 0
		# keep looping infinitely -- the model will stop once the desired number of epochs has been reached
		while epochs < passes:
			for i in np.arange(0, self.numImages, self.batchSize):
				images = self.db["images"][i: i + self.batchSize]
				labels = self.db["labels"][i: i + self.batchSize]

				# check to see if the labels should be binarized
				if self.binarize:
					labels = np_utils.to_categorical(labels, self.classes)		
					# check to see if our preprocessors are not None
				if self.preprocessors is not None:
					#initialize the list of processed images
					procImages = []

					for image in images:
						for p in self.preprocessors:
							image = p.preprocess(image)

						procImages.append(image)
					images = np.array(procImages)
				# apply data augmenator if it is existing						
				if self.aug is not None:
					(images, labels) = next(self.aug.flow(images, labels, batch_size=self.batchSize))
				yield(images, labels)
			epochs += 1

	def close(self):
		self.db.close()
