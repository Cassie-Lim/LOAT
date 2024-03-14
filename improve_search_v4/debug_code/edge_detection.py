''''
测试那种边缘检测效果比较好
'''
import cv2
import numpy as np
import pickle
import os
from time import time

data_peth = "improve_search_v4/debug_data/"

def edge_detect(pickle_path):
    save_path = "improve_search_v4/debug_data/results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(pickle_path, 'rb') as f:
        img = pickle.load(f)
        # 将tensor转为img
        img = img.numpy()[0]
        cv2.imwrite(os.path.join(save_path, 'recep_31.jpg'), img)

        # 用各种方式进行边缘检测

        # 1. 拉普拉斯检测
        start  = time()
        img_lap = cv2.Laplacian(img, cv2.CV_64F)
        img_lap = np.abs(img_lap)
        end = time()
        print("Laplacian time: ", end - start)
        cv2.imwrite(os.path.join(save_path, 'recep_31_lap.jpg'), img_lap)

        # 2. 


        
    
    print('over')

if __name__ == '__main__':
    edge_detect('improve_search_v4/debug_data/recep_31.pkl')
    print("over")