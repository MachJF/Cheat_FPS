import random
import cv2
from numpy import random
from pathlib import Path
import time
import torch

from myDatasets import LoadStreams
from utils.torch_utils import select_device, TracedModel, load_classifier, time_synchronized
from models.experimental import attempt_load
from utils.general import set_logging, apply_classifier, check_imshow, non_max_suppression, scale_coords
import torch.backends.cudnn as cudnn
from utils.plots import plot_one_box


def detect(weights, img_size=640):
    # 初始化设备
    set_logging()
    device_type = '0'  # 默认使用GPU
    device = select_device(device_type)

    # 加载模型
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    # img_size = check_img_size(img_size, s=stride)

    model = TracedModel(model, device, img_size)

    model.half()  # 半精度

    # Second-stage classifier
    classify = False
    if classify:
        model_c = load_classifier(name='resnet101', n=2)  # 初始化
        model_c.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(img_size=img_size, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))  # run once

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img, augment=False)[0]

        # Inference
        with torch.no_grad():
            pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, model_c, img, im0s)

        # Process detections
        for i, det in enumerate(pred):
            p, s, im0, frame = path[i], '%g ' % i, im0s[i].copy(), dataset.count

            p = Path(p)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)


if __name__ == '__main__':
    a = 0
    detect(weights='weights/yolov7.pt')
