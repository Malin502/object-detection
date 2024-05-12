#!/usr/bin/python
# -*- coding: sjis -*-

import xml.etree.ElementTree as ET
import pickle
import numpy as np

voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

dirpath = './VOCdevkit/VOC2012/Annotations/'
datadic = {}

f = open('./VOCdevkit/VOC2012/ImageSets/Main/train.txt','r')
files = f.read().split('\n')

num = 0
for filename in files:
    if filename == '':
        break
    xmlfile = filename + ".xml"
    xml = ET.parse(dirpath + xmlfile).getroot()
    size = xml.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    objdata = []
    for obj in xml.iter('object'):
        difficult = int(obj.find('difficult').text)
        if difficult != 0:
            continue
        num += 1
        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')
        xmin = (float(bbox.find('xmin').text) - 1.0)/float(w)
        ymin = (float(bbox.find('ymin').text) - 1.0)/float(h)
        xmax = (float(bbox.find('xmax').text) - 1.0)/float(w)
        ymax = (float(bbox.find('ymax').text) - 1.0)/float(h)
        if name in voc_classes:
            objdata.append([xmin, ymin, xmax, ymax, float(voc_classes.index(name))])
    if (len(objdata) > 0):
        datadic[filename] = np.array(objdata)
        num += 1
        print(num)

with open('ans.pkl','bw') as fw:
    pickle.dump(datadic,fw)

print('saved ans.pkl, number of data is ',len(datadic))
