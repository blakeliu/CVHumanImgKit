
# pfld
```shell
.\onnx2ncnn.exe D:\cv\face\project\FaceImageKit\weights\pfld\onnx\landmark_lapa_134.onnx D:\cv\face\project\FaceImageKit\weights\pfld\ncnn\landmark_lapa_134.param D:\cv\face\project\FaceImageKit\weights\pfld\ncnn\landmark_lapa_134.bin
```
```shell
.\ncnnoptimize.exe D:\cv\face\project\FaceImageKit\weights\pfld\ncnn\landmark_lapa_134.param D:\cv\face\project\FaceImageKit\weights\pfld\ncnn\landmark_lapa_134.bin D:\cv\face\project\FaceImageKit\weights\pfld\ncnn\landmark_lapa_134-opt.param  D:\cv\face\project\FaceImageKit\weights\pfld\ncnn\landmark_lapa_134-opt.bin 0
```
# scrfd
```shell
.\onnx2ncnn.exe D:\cv\face\project\FaceImageKit\weights\scrfd\onnx\scrfd_500m_bnkps_shape640x640.onnx D:\cv\face\project\FaceImageKit\weights\scrfd\ncnn\scrfd_500m_bnkps_shape640x640.param D:\cv\face\project\FaceImageKit\weights\scrfd\ncnn\scrfd_500m_bnkps_shape640x640.bin
```
```shell
.\ncnnoptimize.exe  D:\cv\face\project\FaceImageKit\weights\scrfd\ncnn\scrfd_500m_bnkps_shape640x640.param D:\cv\face\project\FaceImageKit\weights\scrfd\ncnn\scrfd_500m_bnkps_shape640x640.bin  D:\cv\face\project\FaceImageKit\weights\scrfd\ncnn\scrfd_500m_bnkps_shape640x640-opt.param D:\cv\face\project\FaceImageKit\weights\scrfd\ncnn\scrfd_500m_bnkps_shape640x640-opt.bin 0
```