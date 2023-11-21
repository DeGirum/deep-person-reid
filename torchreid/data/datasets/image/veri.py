from __future__ import division, print_function, absolute_import
import glob
import numpy as np
import os

from torchreid.utils import read_json, write_json

from ..dataset import ImageDataset

class VeRiDataset(ImageDataset):

    # All you need to do here is to generate three lists,
    # which are train, query and gallery.
    # Each list contains tuples of (img_path, carid, camid),
    # Note that
    # - carid and camid should be 0-based.
    # - train, query and gallery share the same camid scope (e.g.
    #   camid=0 in train refers to the same camera as camid=0
    #   in query/gallery).

    def __init__(self, root='', **kwargs):
        PATH_TO_DATASET = "/data/ml-data/vehicle-reid-dataset/" # This is the folder path where you have extracted the zip file and it contains the gallery, query and train images
        FOLDER_PATH, DATASET_TYPES, IMAGES = next(os.walk(f"{PATH_TO_DATASET}")) 
        # print (FOLDER_PATH, DATASET_TYPES, IMAGES)
        training_array = []
        testing_array = []
        query_array = []
        DATASET_TYPES = ["image_train","image_gallery","image_query"]
        for dataset_type in DATASET_TYPES:
            # print (dataset_type)
            _, _, IMAGES = next(os.walk(f"{FOLDER_PATH}/{dataset_type}/"))
            # print ("images ==================",len(IMAGES))
            vehicle_ids = []
            if dataset_type == "image_train" or dataset_type == "image_gallery":
                # print ()

                labelSet = []
                for idx, image in enumerate(IMAGES):
                    pid = int(image.split('.jpg')[0].split('_')[-1])
                    # print (pid)
                    labelSet.append(pid)
                label_dict = {label: index for index, label in enumerate(set(labelSet))}
                # print (label_dict)
                
                for image in IMAGES:
                    cam_id = int(image.split('.jpg')[0].split('_')[-2])
                    # print ("camid ==============",cam_id)
                    # print (int(image.split('.jpg')[0].split('_')[-1]))
                    # if int(image.split('.jpg')[0].split('_')[-1]) == 3:
                    #     print ("key 3")
                    car_id = label_dict[int(image.split('.jpg')[0].split('_')[-1])]
                    vehicle_ids.append(car_id)
                    if dataset_type == "image_gallery":
                        testing_array.append((FOLDER_PATH +  dataset_type + "/" + image, car_id, cam_id))
                    else:
                        training_array.append((FOLDER_PATH +  dataset_type + "/" + image, car_id, cam_id))
                print (len(set(vehicle_ids)))         
            elif dataset_type == "image_query":
                for image in IMAGES:
                    # print (image)
                    cam_id = int(image.split('.jpg')[0].split('_')[-2])
                    # print (int(image.split('.jpg')[0].split('_')[-1]))
                    car_id = label_dict[int(image.split('.jpg')[0].split('_')[-1])]
                    query_array.append((FOLDER_PATH +  dataset_type + "/" + image, car_id, cam_id))

            
        train = training_array
        query = query_array
        gallery = testing_array
        # print (len(set(train)))
        super(VeRiDataset, self).__init__(train, query, gallery, **kwargs)