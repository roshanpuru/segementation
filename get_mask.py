import cv2
import os
def get_segmentation(data):
    img = cv2.imread(os.path.join(ImgDir, data['file_name']))
    height, width = data['height'], data['width']

    colors = [[255, 0, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
            [255, 0, 255]]


    for i, anno in enumerate(data['annotations']):
        bbox = anno['bbox']
        kpt = anno['keypoints']
        segm = anno['segms']
        max_iou = anno['max_iou']

        # img = vistool.draw_bbox(img, bbox, thickness=3, color=colors[i%len(colors)])
        if segm is not None:
            mask = Poly2Mask(segm)
            img = vistool.draw_mask(img, mask, thickness=3, color=colors[i%len(colors)])
        # if kpt is not None:
        #     img = vistool.draw_skeleton(img, kpt, connection=None, colors=colors[i%len(colors)], bbox=bbox)
    return img