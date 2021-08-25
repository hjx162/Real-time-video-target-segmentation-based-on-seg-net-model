# -*- coding=GBK -*-
import cv2 as cv
import os
import tensorflow as tf
import numpy as np


outPutDirName='E:/Python/pythonProject_4/target_tracking_and_detection/result_realtime_seg_save/' # or dog.mp4
frameFrequency=1

# ������ͷ��ȡͼƬ
def video_demo():
    times = 0
    capture = cv.VideoCapture(0, cv.CAP_DSHOW)  # ������ͷ��0��������豸id������ж������ͷ����������������ֵ
    while True:
        times += 1
        res, image = capture.read()  # ��ȡ����ͷ,���ܷ���������������һ��������bool�͵�ret����ֵΪTrue��False��������û�ж���ͼƬ���ڶ���������frame���ǵ�ǰ��ȡһ֡��ͼƬ
        if not res:
            print('not res , not image')
            break
        if times % frameFrequency == 0:
            image = cv.flip(image, 1)  # flip():ͼ��ת����   �ڶ������� С��0: 180����ת������0: ���µߵ�������0: ˮƽ�ߵ�(����ͼ)

            image_1 = image.astype("float32")  # ��������ת�� : ������ͼƬ��int8����ת��Ϊfloat32����
            image_1 = image_1 / 255
            image_1 = cv.resize(image_1, (224, 224))
            image_2 = np.expand_dims(image_1, axis=0)
            image_2 = tf.convert_to_tensor(image_2)

            pred_mask = model.predict(image_2)
            pred_mask = tf.argmax(pred_mask, axis=-1)
            pred_mask = pred_mask[..., tf.newaxis]

            pred_mask_1 = tf.keras.preprocessing.image.array_to_img(pred_mask[0])  # PILת
            pred_mask_2 = np.asarray(pred_mask_1)
            depth_image = cv.applyColorMap(pred_mask_2, cv.COLORMAP_VIRIDIS)
            depth_image = depth_image.astype("float32")
            depth_image = depth_image / 255

            imgs = np.hstack((image_1, depth_image))
            cv.imshow("seg_video", imgs)

            cv.imwrite(outPutDirName + str(times) + '.png', depth_image * 255)
            print(outPutDirName + str(times) + '.png')

            if cv.waitKey(10) & 0xFF == ord('q'):  # �����ע
                 break


video_demo()
cv.destroyAllWindows()