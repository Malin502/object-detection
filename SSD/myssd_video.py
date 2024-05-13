import sys
import torch
import torch.nn as nn
import numpy as np
import cv2

from matplotlib import pyplot as plt
from mynet import SSD
from myfunctions import decode, nms



#Webカメラの設定
DEVICE_ID = 0
WIDTH = 1280 
HEIGHT = 720

#使用するデバイスの設定
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#device = "cuda:0"
print("Using device: " + device)


def main():

    net = SSD(phase = 'test')
    net.load_state_dict(torch.load('ssd1_14.model'))
    #net.to(device)
    
    net.eval()
    
    cap = cv2.VideoCapture(DEVICE_ID)
    
    #フォーマット・解像度の設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    # フォーマット・解像度・FPSの取得 
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps:{}　width:{}　height:{}".format(fps, width, height))
    
    
    while True:
        
        #カメラから画像を取得
        _, frame = cap.read()
        if(frame is None):
            continue
        
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame = cv2.resize(frame, (300, 300)).astype(np.float32)
        frame_tensor = frame - (104.0, 117.0, 123.0)
        frame_tensor = frame[:, :, ::-1].copy() #BGR -> RGB
        frame_tensor = frame.transpose(2, 0, 1)
        #print(frame.shape)
        frame_tensor = torch.from_numpy(frame_tensor).unsqueeze(0)
        
        with torch.no_grad():
            y = net(frame_tensor)
            
        labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                  'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                  'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
            
        colors = np.random.uniform(0, 255, size = (len(labels), 3))
        detection = y.data
        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
        
        for i in range(detection.size(1)):
            j = 0
            
            while detection[0, i, j, 0] >= 0.6:
                score = detection[0, i, j, 0]
                label_name = labels[i-1]
                display_txt = '%s: %.2f'%(label_name, score)
                pt = (detection[0, i, j, 1:] * scale).cpu().numpy()
                color = colors[i] #クラスごとに色が決まっている
                cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), color, 2)
                cv2.putText(frame, display_txt, (int(pt[0]), int(pt[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        j += 1
        
        frame = frame[:, :, ::-1]
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
main()