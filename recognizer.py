#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.path.abspath('./data'))

from sklearn.externals import joblib
cl = joblib.load('kmeans40k.pkl')
import pickle
f = open('labeled.map', 'r')
label_map = pickle.load(f)
f.close()

import cv2
import numpy as np

import preprocess

def get_captcha(captcha_str):
    nparr = np.fromstring(captcha_str, np.uint8)
    ptcha = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
    captcha.shape = -1,
    captcha = (captcha[::3].astype(np.int) + captcha[1::3].astype(np.int) + captcha[2::3].astype(np.int)) / 3
    captcha = (255 - captcha).astype(np.uint8)
    return captcha

def clean_captcha(captcha):
    captcha[captcha < 127] = 0
    captcha[captcha >= 127] = 255
    captcha.shape = 100, 210
    captcha = (captcha[::2, ::2] + captcha[1::2, ::2] + captcha[::2, 1::2] + captcha[1::2, 1::2]) / 4

    c_line = preprocess.del_line(captcha)
    cleaned = []
    for p in preprocess.split_pic(c_line):
        c_p = c_line[:, p[0]:p[1]]
        y1, y2 = preprocess.split_y(c_p)
        cleaned.append(np.reshape(preprocess.regularize(c_p[y1:y2, :]), (-1,)))
    return cleaned

def predict_captcha(char, label_map, cl):
    assert len(char.shape) < 3
    if len(char.shape) == 1:
        char.shape = int(char.shape[0]**.5), -1
    char = (char[::2, ::2] + char[1::2, ::2] + char[::2, 1::2] + char[1::2, 1::2]) / 4
    char.shape = -1,
    return label_map[cl.predict(char.astype(float))[0]]

def predict(chars):
    return ''.join([predict_captcha(char, label_map, cl) for char in chars])
