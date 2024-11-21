import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import os
import math
from PIL import Image
from datetime import datetime
import shutil
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def search_jpg_and_make_dataset(folder_path):
    folder_list=[]
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            sub_folder_path = os.path.join(root, dir)
            # 检查该子文件夹下是否有jpg文件
            has_jpg = any(file.endswith('.jpg') for file in os.listdir(sub_folder_path))
            if has_jpg:
                folder_list.append(sub_folder_path)
    return folder_list
def detect(opt):
    group_directory={}
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # 获取当前日期
    current_date = datetime.now().date()
    # 将日期格式化为字符串
    date_str = current_date.strftime('%Y-%m-%d')
    # 初始化
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # 加载模型
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if trace:
        model = TracedModel(model, device, opt.img_size)
    if half:
        model.half()  # to FP16
    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    sub_source_list=search_jpg_and_make_dataset(source)
    for sub_source in sub_source_list:
        number=0
        save_dir = sub_source.replace(opt.source, opt.project)
        print(opt.project,save_dir)
        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(sub_source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(sub_source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
            else:
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    # print(det)
                    if webcam:  # batch_size >= 1
                        p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                    else:
                        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    # print(f"帧数：{frame}")
                    
                    image_list=[]
                    area_list=[]
                    img_conf=[]
                    number+=len(det)
        group_directory[f'{sub_source}']=number
        print(f'{sub_source}:{number}')
    # 筛选出每个日期下数值最大的项
    max_dict = {}
    for path, value in group_directory.items():
        # 从路径中提取日期
        date = path.split('/')[2]
        # 如果日期还未在max_dict中，或当前项的数值大于max_dict中相同日期的数值，则更新max_dict
        if date not in max_dict or value > max_dict[date][1]:
            max_dict[date] = (path, value)
    print(max_dict)
    # 目标文件夹路径
    target_folder = 'farm_target_reid'
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    # 将筛选出的文件夹移动到目标文件夹
    for date, (path, value) in max_dict.items():
        destination = f"{target_folder}/{date}_{value}"
        if not os.path.exists(destination):
            os.makedirs(destination)
        shutil.move(path, destination)
        print(f"Moved {path} to {destination}")


def run_detection(weights='yolov7_best.pt', source='./18test', img_size=640, conf_thres=0.25, iou_thres=0.45, device='', 
                    view_img=False, save_txt=False, save_conf=False, nosave=False, classes=19, agnostic_nms=False, augment=False,
                    update=False, project='runs/detect', name='', exist_ok=False, no_trace=False):
    """
    Run detection with specified parameters.
    
    Args:
    - weights (str): Path to the model weights.
    - source (str): Source for the input. Can be a path to image/video file(s) or directory, or '0' for webcam.
    - img_size (int): Inference size in pixels.
    - conf_thres (float): Object confidence threshold.
    - iou_thres (float): IOU threshold for NMS.
    - device (str): CUDA device, e.g., '0' or 'cpu'.
    - view_img (bool): If True, display results.
    - save_txt (bool): If True, save results to *.txt files.
    - save_conf (bool): If True, save confidences in --save-txt labels.
    - nosave (bool): If True, do not save images/videos.
    - classes (int or list of int): Filter by class; single class (int) or multiple classes (list).
    - agnostic_nms (bool): If True, perform class-agnostic NMS.
    - augment (bool): If True, perform augmented inference.
    - update (bool): If True, update all models.
    - project (str): Path to save results to project/name.
    - name (str): Name for saved results within the project directory.
    - exist_ok (bool): If True, existing project/name is okay, do not increment.
    - no_trace (bool): If True, don't trace the model.
    """
    # Convert classes to list if it is a single integer
    if isinstance(classes, int):
        classes = [classes]

    # Setup argparse values
    opt = argparse.Namespace(weights=weights, source=source, img_size=img_size, conf_thres=conf_thres, iou_thres=iou_thres, 
                            device=device, view_img=view_img, save_txt=save_txt, save_conf=save_conf, nosave=nosave, classes=classes, 
                            agnostic_nms=agnostic_nms, augment=augment, update=update, project=project, name=name, exist_ok=exist_ok, no_trace=no_trace)
    print(opt)

    with torch.no_grad():
        if opt.update:  # Update all models to fix SourceChangeWarning
            for weight in ['yolov7_best.pt']:
                detect(opt)  # Assuming detect() is defined elsewhere
                strip_optimizer(weight)  # Assuming strip_optimizer() is defined elsewhere
        else:
            detect(opt) # Run detection
run_detection(source="./farm_reid", project="processed_cow_image")