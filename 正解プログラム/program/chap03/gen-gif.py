#!/usr/bin/python
# -*- coding: sjis -*-

from PIL import Image
import glob
 
files = sorted(glob.glob('detect_dir/*.png'))
images = [ Image.open(f) for f in files ]
images[0].save('sample-video-ssd.gif', save_all=True, append_images=images[1:], duration=200, loop=0)

