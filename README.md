#FaceImageKit

Python实现的人脸图像相关算法Pipeline

# Install
python >= 3.8  
```shell
pip install -r requirements.txt
```
## onnxruntime env for mmdeploy
- Linux-x86_64
[mmdeploy安装地址](https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/get_started.md)
```shell
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.8.1
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```
或者


```shell
wget https://github.com/open-mmlab/mmdeploy/releases/download/v1.1.0/mmdeploy-1.1.0-linux-x86_64.tar.gz
tar -zxvf mmdeploy-1.1.0-linux-x86_64.tar.gz
export ONNXRUNTIME_DIR=$(pwd)/mmdeploy-1.1.0-linux-x86_64/thirdparty/onnxruntime/lib
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```
- Win10
[mmdeploy安装地址](https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/02-how-to-run/prebuilt_package_windows.md)

https://github.com/microsoft/onnxruntime/releases/tag/v1.8.1
```shell
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-win-x64-1.8.1.zip
```
或者

```shell
wget https://github.com/open-mmlab/mmdeploy/releases/download/v1.1.0/mmdeploy-1.1.0-windows-amd64.zip
```
设置lib目录到系统环境变量

# Models
## Face Dtection
### scrfd
Models accuracy on WiderFace benchmark:
| Model               |  Easy   |   Medium   | Hard  |
|:--------------------|:-------:|:----------:|:-----:|
| scrfd_10g_gnkps     |  95.51  |   94.12    | 82.14 |
| scrfd_2.5g_gnkps    |  93.57  |   91.70    | 76.08 |
| scrfd_500m_gnkps    |  88.70  |   86.11    | 63.57 |

来源https://github.com/SthPhoenix/InsightFace-REST/    

**runtime**
+ [x] onnxruntime(cpu)
+ [x] ncnn
+ [ ] TensorRT
+ [ ] mmdeploy

## Face Landmark
datasets: Lapa134(Lapa106 + 28)
### PFLD
项目来源：https://gold-one.coding.net/p/BeautyFace/d/TFace_Landmark/git
模型输出大小: $134\times 2$
**runtime**
+ [x] onnxruntime(cpu)
+ [x] ncnn
+ [ ] TensorRT
+ [ ] mmdeploy

### rtmface
项目来源 https://blakeliu.coding.net/p/face/d/mmpose/git/tree/master/configs/face_2d_keypoint/rtmpose/lapa
模型输出大小: $134\times 2$

| Model               |  NME   |
|:--------------------|:-------:|
| rtmpose-m-ort-lapa134  |  0.0288  |
| rtmpose-s-ort-lapa134  |  0.0258  |
+ [x] onnxruntime(cpu)
+ [ ] ncnn
+ [ ] TensorRT
+ [x] mmdeploy

## Face Segmentation
### ppliteseg
项目来源：https://github.com/tfrbt/FaceSeg.git

- label map 类别数量32
  
| id | class  |
|:-----|:---:|
|0  | 'background' |
|1  | 'skin' |
|2  | 'cheek' |
|3  | 'chin' |
|4  | 'ear' |
|5  | 'helix' |
|6  | 'lobule' |
|7  | 'bottom_lid' |
|8  | 'pupil' |
|9  | 'iris' |
|10 |  'sclera' |
|11 |  'tear_duct' |
|12 |  'top_lid' |
|13 |  'eyebrow' |
|14 |  'forhead' |
|15 |  'frown' |
|16 |  'hair' |
|17 |  'temple' |
|18 |  'jaw' |
|19 |  'beard' |
|20 |  'inferior_lip' |
|21 |  'oral comisure' |
|22 |  'superior_lip' |
|23 |  'teeth' |
|24 |  'neck' |
|25 |  'nose' |
|26 |  'ala_nose' |
|27 |  'bridge' |
|28 |  'nose_tip' |
|29 |  'nostril' |
|30 |  'DU26' |
|31 |  'sideburns' |

| Model               | val IOU   |val Dice|
|:--------------------|:-------:|:-------:|
| face_seg_ppliteseg_t  |  0.727  |0.834|

+ [x] onnxruntime(cpu)
+ [x] ncnn
+ [ ] TensorRT
+ [ ] mmdeploy

- label map 类别数量12

| id | class  |
|:-----|:---:|
|0  | "background"  # 0.背景 |
|1  | "skin"  # 1.皮肤 |
|2  | "eye"  # 2.眼睛 |
|3  | "pupil"  # 3.瞳孔 |
|4  | "bottom_lid", # 4.下眼皮 |
|5  | "top_lid", # 5.上眼皮 |
|6  | "eyebrow"  # 6.眉毛 |
|7  | "hair"  # 7.头发 |
|8  | "superior_lip"  # 8上嘴唇 |
|9  | "teeth"  # 9.牙齿 |
|10 |  "inferior_lip" # 10.下嘴唇 |
|11 |  "nose" # 11.鼻子 |

| Model               | val IOU |val Dice|
|:--------------------|:-------:|:-------:|
| face12_seg_pplitesegb  |  0.774  |0.869|

+ [x] onnxruntime(cpu)
+ [x] ncnn
+ [ ] TensorRT
+ [ ] mmdeploy