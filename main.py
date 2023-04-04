import torch
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import sys
import numpy as np
import os, json, cv2, random
from collections import defaultdict

# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
sys.path.insert(0, 'Detic/')
sys.path.insert(0, 'Detic/third_party/CenterNet2/')
from detic.modeling.text.text_encoder import build_text_encoder

import pickle
from typing import List, Dict, Tuple
import open3d as o3d
# Detic libraries 
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from sklearn.cluster import DBSCAN
original_working_directory = os.getcwd()

# Set the path to the data directory
DATA_DIR = "/home/kishore/workspace/fewshot/data/230321_examples.pkl"

# Define a function to get the object detection predictor
def get_predictor():
    # Build the detector and download our pretrained weights
    cfg = get_cfg()
    # Add the Centernet and Detic configurations to the model configuration
    add_centernet_config(cfg)
    add_detic_config(cfg)
    # Merge the model configuration from the YAML file
    cfg.merge_from_file("Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    # Set the pretrained weights for the model
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3 # set threshold for this model
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
    # Uncomment the following line to use CPU-only mode
    # cfg.MODEL.DEVICE='cpu'
    predictor = DefaultPredictor(cfg)
    return predictor

# Define a function to get the CLIP embeddings of the vocabulary
def get_clip_embeddings(vocabulary, prompt='a '):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

# Define a function to load the data from the data directory
def load_data():
    with open(DATA_DIR, 'rb') as f:
        data = pickle.load(f)
    return data

# Define a function to initialize the vocabulary for the object classes
def initialize_vocab(predictor):
    # Set the vocabulary to custom
    vocabulary = 'custom'
    metadata = MetadataCatalog.get("__unused10")  # Modify the name when you change the thing classes
    # Set the thing classes that the model can detect
    metadata.thing_classes = ['cup', 'bowl', 'mug', 'bottle', 'cardboard', 'Tripod', 'Baseball bat' , 'Lamp', 'Mug Rack']
    # Get the CLIP embeddings of the thing classes
    classifier = get_clip_embeddings(metadata.thing_classes)
    num_classes = len(metadata.thing_classes)
    # Reset the classifier for testing
    reset_cls_test(predictor.model, classifier, num_classes)
    # Reset the visualization threshold
    output_score_threshold = 0.3
    for cascade_stages in range(len(predictor.model.roi_heads.box_predictor)):
        predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold
    return metadata

# Define a function to visualize the output of the object detection
def visualize(img):
    out_im = cv2.cvtColor(img.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB
    cv2.imshow("output",out_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Define a function to perform object detection on the images and visualize the results
def predict_and_visualize(data,metadata, predictor):

    for item in data:
        images = item['images']
        depths = item['depths']
        clouds = item['clouds']
        for i in range(len(images)):
            image = images[i]
            depth = depths[i]
            cloud = clouds[i]
            # Reshape the point cloud and convert from BGR to RGB
            cloud_reshaped = cloud.reshape(720, 1280, 3)[:, :, ::-1] * 3

            # Perform object detection on the image using the Detectron2 model
            outputs = predictor(image)
            instances = outputs["instances"].to("cpu")
            detections = instances.pred_boxes.tensor.numpy()
            classes = instances.pred_classes.numpy()
            v = Visualizer(image[:, :, ::-1], metadata)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            # Visualize the output
            visualize(out)

if __name__ == '__main__':
    os.chdir("Detic")
    predictor = get_predictor()
    data = load_data()
    metadata = initialize_vocab(predictor)
    predict_and_visualize(data,metadata, predictor)

   
