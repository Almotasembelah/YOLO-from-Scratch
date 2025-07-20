import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F

import time
from functools import wraps

COLORS = {
    'Bus': (255, 0, 0),      # Red
    'Truck': (0, 255, 0),      # Green
}

def show_im(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def draw_boxes(image, boxes, labels=None, normalized=True, scores=None, show=True):
    from torchvision.utils import draw_bounding_boxes
    from torchvision.ops import box_convert
    from PIL import ImageDraw, ImageFont
    _, h, w = image.size()

    colors = (0, 255, 0)
    if labels:
        colors = [COLORS[label] for label in labels]

    boxes = torch.tensor([[x*w, y*h, x1*w, y1*h] for x, y, x1, y1 in boxes]) if normalized else boxes
    boxes = box_convert(boxes, 'cxcywh', 'xyxy')
    image = (image*255).type(torch.uint8)
    img = draw_bounding_boxes(image, boxes, colors=colors, width=3)
    if scores is None:
        scores = [None] * len(boxes)
    if labels:
        img_pil = F.to_pil_image(img.cpu())
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", size=20)
            font_bold = ImageFont.truetype("arialbd.ttf", size=20)
        except Exception:
            font = ImageFont.load_default()
            font_bold = font
            try:
                font = ImageFont.truetype("dejavu/DejaVuSans.ttf", size=20)
                font_bold = ImageFont.truetype("dejavu/DejaVuSans-Bold.ttf", size=20)
            except Exception:
                pass
        for label, bbox, score in zip(labels, boxes, scores):
            text = label if score is None else f'{label} {score*100:.2f}%'
            x1, y1, x2, y2 = bbox.tolist()
            bbox_text = font.getbbox(text)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            margin = 10
            # Clamp coordinates to stay within image boundaries
            rect_x1 = max(x1, 0)
            rect_y1 = max(y1 - text_height - margin, 0)
            rect_x2 = min(x1 + text_width + 2 * margin, w)
            rect_y2 = min(y1, h)
            # Ensure the rectangle fully covers the text area
            if rect_y2 - rect_y1 < text_height:
                rect_y2 = rect_y1 + text_height
            if rect_x2 - rect_x1 < text_width + 2 * margin:
                rect_x2 = rect_x1 + text_width + 2 * margin
            rect_x2 = min(rect_x2, w)
            rect_y2 = min(rect_y2, h)
            # Draw rectangle for background
            draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill=COLORS[label])
            # Draw text inside the rectangle, centered vertically
            text_x = rect_x1 + margin
            text_y = rect_y1 + (rect_y2 - rect_y1 - text_height) // 2 if rect_y2 - rect_y1 - text_height > 0 else rect_y1
            draw.text((text_x, text_y), text, fill='black', font=font_bold)
        img = F.to_tensor(img_pil)
    if show:
        show_im(img)
    else:
        return img

def nms(prediction:torch.Tensor, iou_thresh=0.5, conf_thresh=0.5, nc=2, max_det=300, max_nms=30000, classes=None, max_wh=7680):
    import torchvision
    from torchvision.ops import box_convert

    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thresh  # candidates
    xinds = torch.stack([torch.arange(len(i), device=prediction.device) for i in xc])[..., None]  # to track idxs
    
    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    prediction = torch.cat((box_convert(prediction[..., :4], 'cxcywh', 'xyxy'), prediction[..., 4:]), dim=-1)

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs  # to store the kept idxs
    for xi, (x, xk) in enumerate(zip(prediction, xinds)):  # image index, (preds, preds indices)
        filt = xc[xi]  # confidence
        x, xk = x[filt], xk[filt]

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        # best class only
        conf, j = cls.max(1, keepdim=True)
        filt = conf.view(-1) > conf_thresh
        x = torch.cat((box, conf, j.float(), mask), 1)[filt]
        xk = xk[filt]

        # Filter by class
        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x, xk = x[filt], xk[filt]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            filt = x[:, 4].argsort(descending=True)[:max_nms]  # sort by confidence and remove excess boxes
            x, xk = x[filt], xk[filt]

        # Batched NMS
        c = x[:, 5:6] * (max_wh)  # classes
        scores = x[:, 4]  # scores:
        boxes = x[:, :4] + c  # boxes (offset by class)
        i = torchvision.ops.nms(boxes, scores, iou_thresh)  # NMS

        i = i[:max_det]  # limit detections
        output[xi], keepi[xi] = x[i], xk[i].reshape(-1)

    return output

def speed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱️ Time taken: {end - start:.4f} seconds")
        return result
    return wrapper