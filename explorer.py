#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.path.abspath('.'))

import numpy as np
import pylab as pl
import time

import recognizer

captchas = np.load('data/raw_captchas.npy')

for c in captchas:
    pl.figure(1)
    start = time.time()
    cleaned = recognizer.clean_captcha(c)
    y = recognizer.predict(cleaned)
    print('%f secs' % (time.time() - start))
    pl.imshow(c, cmap=pl.cm.Greys)
    pl.title(y)
    pl.show()
    if raw_input() == 'q':
        pl.close('all')
        break
    pl.close(1)
