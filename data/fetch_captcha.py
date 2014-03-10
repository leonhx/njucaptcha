#!/usr/bin/python
# -*- coding: utf-8 -*-

import urllib2
import cv2
import numpy as np

captchas = []
N = 10000

while N > 0:
    page = urllib2.urlopen('http://cer.nju.edu.cn/amserver/verify/image.jsp')
    captcha_str = page.read()
    nparr = np.fromstring(captcha_str, np.uint8)

    captcha = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
    captcha.shape = -1,

    captcha = captcha[::3].astype(np.int) + captcha[1::3].astype(np.int) + \
                captcha[2::3].astype(np.int)
    captcha = captcha/3
    captcha = (255 - captcha).astype(np.uint8)

    captchas.append(captcha)
    N -= 1

captchas = np.array(captchas)
np.save('raw_captchas.npy', captchas)
