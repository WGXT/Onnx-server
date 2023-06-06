from fastapi import FastAPI, UploadFile, File

from io import BytesIO
from PIL import Image,ImageDraw
from utils.operation import YOLO
from detect import detect

#  FastAPI 框架编写的简单应用程序，其中包含两个端点，
# 分别使用装饰器 @app.get() 和 @app.post() 进行定义
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/detect/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()  # 接收浏览器上传的图片
    im1 = BytesIO(contents)  # 将数据流转换成二进制文件存在内存中

    # 返回结果
    return detect(onnx_path='Onnx-Model/detect-Occ.onnx', img_path=im1, show=False)
