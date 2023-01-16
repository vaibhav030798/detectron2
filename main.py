from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import matplotlib.pyplot as plt
import torch
import cv2

##if __name__ == "__main__":

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

def cv2_imshow(im):

    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    re_im = cv2.resize(im, (512,512))
    cv2.imshow('Result', re_im)
    cv2.waitKey(0)
##    plt.figure()
##    plt.imshow(im)
##    plt.axis('off')

def cyc_dicts(img_dir):
    json_file = os.path.join(img_dir, "Rscyc_json.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
    
        annos = v["regions"]
        objs = []
        # for _, anno in annos.items():
        #     assert not anno["region_attributes"]
        #     anno = anno["shape_attributes"]
        #     px = anno["all_points_x"]
        #     py = anno["all_points_y"]
        #     poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        #     poly = [p for x in poly for p in x]


        for anno in annos:
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts





for d in ["train", "val"]:
    DatasetCatalog.register("cylinder_" + d, lambda d=d: cyc_dicts("cylinder_datasets/" + d))
    MetadataCatalog.get("cylinder_" + d).set(thing_classes=["full body","Dent"])




balloon_metadata = MetadataCatalog.get("cylinder_train")



class Metadata:
    def get(self, _):
        return ['full body','Dent'] #your class labels


# To verify the data loading is correct, let's visualize the annotations of randomly selected samples in the training set:



import matplotlib.pyplot as plt
dataset_dicts = cyc_dicts("cylinder_datasets/train")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], Metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    #cv2_imshow(out.get_image()[:, :, ::-1])



from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.MODEL.DEVICE = "cuda"

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("cylinder_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES =  2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
#trainer.train()


# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold

predictor = DefaultPredictor(cfg)


#import parallelTestModule
from detectron2.utils.visualizer import ColorMode
dataset_dicts = cyc_dicts("cylinder_datasets/val")

import glob

path = r'F:\Mask-TF2\test\*.jpg'

#for d in random.sample(dataset_dicts, 100):

for file in glob.glob(path):
    im = cv2.imread(file)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                Metadata, 
                scale=0.5   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])


# if __name__ == '__main__':    
#     extractor = parallelTestModule.ParallelExtractor()
#     extractor.runInParallel(numProcesses=2, numThreads=4)



# if __name__ == '__main__':
#     run()
