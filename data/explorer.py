#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl

captchas = np.load('raw_captchas.npy')
# captchas[captchas < 127] = 0
# captchas[captchas >= 127] = 255
captchas.shape = len(captchas), 100, 210

for c in captchas:
    pl.figure(1)
    pl.imshow(c, cmap=pl.cm.Greys)
    pl.show()
    if raw_input() == 'q':
        pl.close(1)
        break
    pl.close(1)
