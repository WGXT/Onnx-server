from PIL import Image,ImageDraw
from utils.operation import YOLO

def detect(onnx_path='Onnx-Model/detect-Occ.onnx',img_path='img/demo.jpg',show=True):
    '''
    检测目标，返回目标所在坐标如：
    {'crop': [57, 390, 207, 882], 'classes': 'person'},...]
    :param onnx_path:onnx:模型路径
    :param img_path:检测用的图片
    :param show:是否展示
    :return:
    '''
    yolo = YOLO(onnx_path=onnx_path)
    det_obj = yolo.decect(img_path)

    # 检测结果
    print (det_obj)

    # 画出检测框框(如果需要)
    if show:
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)

        for i in range(len(det_obj)):
            draw.rectangle(det_obj[i]['crop'],width=3)
        img.show()  # 展示

    return det_obj

if __name__ == "__main__":
    detect()