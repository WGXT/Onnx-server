import numpy as np

np.set_printoptions(precision=4)

def rescale_boxes(boxes, current_dim, original_shape):
    """ 
    将边界框重新缩放为原始形状 
    """
    orig_h, orig_w = original_shape
    # 计算添加的填充量
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # 删除填充后的图像高度和宽度
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # 将边界框重新缩放到原始图像的尺寸
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes

def tag_images(imgs, img_detections, img_size, classes):
    """
    函数循环遍历每张图像及其对应的检测结果，将边界框缩放到原始图像大小，
    并为每个检测出的对象生成一个字典，包含其裁剪坐标和预测类别。
    如果未在某个图像中检测到任何对象，则打印出"识别失败"并继续处理下一张图像
    """
    imgs = [imgs]
    results = []
    for img_i, (img, detections) in enumerate(zip(imgs, img_detections)):
        if detections is not None:
            # 将框重新缩放为原始图像
            detections = rescale_boxes(detections, img_size, img.shape[:2])
            for x1, y1, x2, y2, conf, cls_pred in detections:
                results.append(
                    {
                        "crop": [int(i) for i in (x1, y1, x2, y2)],
                        "classes": classes[int(cls_pred)]
                    }
                )
        else:
            print("识别失败")
    return results

def xywh2xyxy(x):
    '''
    将形如 [x, y, w, h] 的 nx4 个矩形框转换为形如 [x1, y1, x2, y2] 的矩形框，
    其中 x1,y1 表示左上角坐标，x2,y2 表示右下角坐标。
    '''
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def nms(dets, scores, thresh):
    '''
    非极大值抑制函数(Non-Maximum Suppression)
    目的：目标检测领域，当使用某个算法得到多个候选框时，为了避免重复检测和提高检测精度，需要对这些候选框进行筛选
    '''
    # x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]  # xmin
    y1 = dets[:, 1]  # ymin
    x2 = dets[:, 2]  # xmax
    y2 = dets[:, 3]  # ymax
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # argsort()返回数组值从小到大的索引值
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:  # 还有数据
        i = order[0]
        keep.append(i)
        if order.size == 1: break

        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交框的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        IOU = inter / (areas[i] + areas[order[1:]] - inter)
        left_index = (np.where(IOU <= thresh))[0]
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[left_index + 1]

    return np.array(keep)