#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl

import time

captchas = np.load('clean_chars.npy')
captchas.shape = -1, 40, 40

for c in captchas:
    pl.figure(1)
    pl.imshow(c, cmap=pl.cm.Greys)
    pl.show()
    time.sleep(1)
    pl.close(1)
