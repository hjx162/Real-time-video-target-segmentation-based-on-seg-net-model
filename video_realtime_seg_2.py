# -*- coding=GBK -*-
import cv2 as cv
import os
import tensorflow as tf
import numpy as np


outPutDirName='E:/Python/pythonProject_4/target_tracking_and_detection/result_realtime_seg_save/' # or dog.mp4
frameFrequency=1

# 打开摄像头获取图片
def video_demo():
    times = 0
    capture = cv.VideoCapture(0, cv.CAP_DSHOW)  # 打开摄像头，0代表的是设备id，如果有多个摄像头，可以设置其他数值
    while True:
        times += 1
        res, image = capture.read()  # 读取摄像头,它能返回两个参数，第一个参数是bool型的ret，其值为True或False，代表有没有读到图片；第二个参数是frame，是当前截取一帧的图片
        if not res:
            print('not res , not image')
            break
        if times % frameFrequency == 0:
            image = cv.flip(image, 1)  # flip():图像翻转函数   第二个参数 小于0: 180°旋转，等于0: 上下颠倒，大于0: 水平颠倒(镜像图)

            image_1 = image.astype("float32")  # 数据类型转换 : 将读入图片的int8类型转换为float32类型
            image_1 = image_1 / 255
            image_1 = cv.resize(image_1, (224, 224))
            image_2 = np.expand_dims(image_1, axis=0)
            image_2 = tf.convert_to_tensor(image_2)

            pred_mask = model.predict(image_2)
            pred_mask = tf.argmax(pred_mask, axis=-1)
            pred_mask = pred_mask[..., tf.newaxis]

            pred_mask_1 = tf.keras.preprocessing.image.array_to_img(pred_mask[0])  # PIL转
            pred_mask_2 = np.asarray(pred_mask_1)
            depth_image = cv.applyColorMap(pred_mask_2, cv.COLORMAP_VIRIDIS)
            depth_image = depth_image.astype("float32")
            depth_image = depth_image / 255

            imgs = np.hstack((image_1, depth_image))
            cv.imshow("seg_video", imgs)

            cv.imwrite(outPutDirName + str(times) + '.png', depth_image * 255)
            print(outPutDirName + str(times) + '.png')

            if cv.waitKey(10) & 0xFF == ord('q'):  # 详见备注
                 break


video_demo()
cv.destroyAllWindows()