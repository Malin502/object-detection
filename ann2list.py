import xml.etree.ElementTree as ET
import pickle
import numpy as np


voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat',
                'Bottle', 'Bus', 'Car', 'Cat', 'Chair',
                'Cow', 'Diningtable', 'Dog', 'Horse',
                'Motorbike', 'Person', 'Pottedplant',
                'Sheep', 'Sofa', 'Train', 'Tvmonitor']

dirpath = './VOCdevkit/VOC2007/anotations/'
datadic = {}

f = open('./VOCdevkit/VOC2007/ImageSets/Main/trainval.txt','r')
files = f.read().split('\n')

num = 0
for filename in files:
    if filename == '':
        break
    xmlfile = filename + '.xml'
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
            
        bbox = obj
        xmin = (float(bbox.find('xmin').text) - 1) / w
        ymin = (float(bbox.find('ymin').text) - 1) / h
        xmax = (float(bbox.find('xmax').text) - 1) / w
        ymax = (float(bbox.find('ymax').text) - 1) / h
        
        if name in voc_classes:
            objdata.append([xmin, ymin, xmax, ymax, voc_classes.index(name)])
            num += 1
            print(num)
            
        with open('ans.pkl','bw') as fw:
            pickle.dump(datadic, f)
            
        print('save ans.pkl, number of data is', len(datadic))
        
        