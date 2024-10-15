
### Yolov8实现车牌检测

训练YOLOV8模型，对于车牌数据集CCPD进行预测，并导出模型可以支持直接调用。该实现脚本储存在 Yolov8文件夹下。


环境配置： 

ultralytics-8.2.70



```python
# 利用训练好的模型Predict   

from ultralytics import YOLO
model_plate=YOLO(model='./runs/segment/train3/weights/best.pt'
               ,task="segment")
results=model_plate("./images/train/00395833333333-90_268-243&427_357&462-357&460_243&462_244&428_356&427-0_0_3_24_24_24_32_33-145-17.jpg")

for result in results:
    orig_img=result.orig_img
    boxes=result.boxes
    masks=result.masks

import matplotlib.pyplot as plt
import cv2
plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
plt.show()


# 数据处理见脚本：data_process.ipynb

# 数据训练见脚本：yolov8_plate_train.ipynb
``` 
![Yolov8_Result](yolov8.PNG)
