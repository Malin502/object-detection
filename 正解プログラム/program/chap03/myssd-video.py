#!/usr/bin/python
# -*- coding: sjis -*-

from imutils.video import FPS, WebcamVideoStream

stream = WebcamVideoStream(src=0).start()  # default camera
fps = FPS().start()
while True:
    frame = stream.read()  # (480, 640, 3)
    key = cv2.waitKey(1) & 0xFF
    fps.update()
    frame = predict(frame) # 物体検出を行う BBox 付き画像
    if key == ord('p'):  # pause
        while True:
            key2 = cv2.waitKey(1) or 0xff
            cv2.imshow('frame', frame)
            if key2 == ord('p'):  # resume
                break
    cv2.imshow('frame', frame)
    if key == 27:  # exit
        break
fps.stop()
