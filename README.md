# Onnx-server
 使用yolo分类和检测转化的onnx模型，实现消防通道占用检测的服务部署

## 项目功能
 搭建onnx检测平台，并使用onnx格式模型文件检测目标，可本地使用，也可作为服务的方式，通过API向外部提供服务。
## Onnx介绍
 ONNX 是一种用于表示机器学习的开放格式 模型。ONNX 定义了一组通用运算符（机器学习和深度学习模型的构建基块）和通用文件格式，使 AI 开发人员能够使用具有各种框架、工具、运行时和编译器的模型。  
官网地址：https://onnx.ai/  
可视化onnx模型网络结构：https://netron.app/

## 安装方式


## 服务部署
```
uvicorn main:app --reload --host 0.0.0.0
```