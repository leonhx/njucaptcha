#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
import pylab as pl

WHITE = 0
BLACK = 255

captchas = numpy.load('raw_captchas.npy')

captchas[captchas < 127] = 0
captchas[captchas >= 127] = 255

captchas.shape = len(captchas), 100, 210
captchas = (captchas[:, ::2, ::2] + captchas[:, 1::2, ::2] + captchas[:, ::2, 1::2] + captchas[:, 1::2, 1::2]) / 4

WIDTH = 3
Y1 = 0
Y2 = 99
X_L = 0
X_R = 49

def get_pic(x1, x2):
    k = 1. * (x2 - x1) / (99 - 0)
    b = 1. * (x1*99 - x2*0) / (99 - 0)
    def f(y):
        return k * y + b
    pic = numpy.zeros((50, 105), dtype=int)
    points = [(round(f(y)), y) for y in range(0, 99+1)]
    for x, y in points:
        for i in range(0, 3):
            if 0 <= x+i <= 49:
                pic[x+i, y] = 255
    return pic

def optimal_result(captcha, lines):
    min_area = 5250
    best_result = None
    for line in lines:
        t = captcha - get_pic(*line)
        t = numpy.where(t>0, 1, 0)
        t_area = t.sum()
        if t_area < min_area:
            min_area = t_area
            best_result = t
    if best_result is None:
        raise AssertionError()
    return best_result * 255

def del_dot(captcha):
    """invoke this after eliminating lines"""
    captcha = numpy.c_[numpy.zeros(len(captcha), dtype=int), captcha, numpy.zeros(len(captcha), dtype=int)]
    captcha = numpy.vstack((numpy.zeros(len(captcha[0]), dtype=int), captcha, numpy.zeros(len(captcha[0]), dtype=int)))
    for i in range(1, 49+2):
        for j in range(1, 99+2):
            if captcha[i, j] > 0:
                if captcha[(i-1):(i+2), (j-1):(j+2)].sum() == captcha[i, j]:
                    captcha[i, j] = 0
    return captcha[1:-1, 1:-1]

def split_pic(captcha):
    """
    captcha should have been line-eliminated and dot-elimated
    return 4 lenth-2 tuples to indicate y-coordinate 4 characters in captcha
    """
    intervals = []
    l = r = 0
    IN = False
    for j in range(len(captcha[0])):
        if captcha[:, j].sum() < 255*3:
            if IN:
                r = j
                intervals.append((l, r))
                IN = False
        else:
            if not IN:
                l = j
                IN = True
    if IN:
        r = 105
        intervals.append((l, r))
    while len(intervals) > 4:
        in_len = numpy.array([inte[1]-inte[0] for inte in intervals])
        min_i = numpy.argmin(in_len)
        if min_i == 0:
            intervals = combine_right(intervals, min_i)
        elif min_i == len(intervals)-1:
            intervals = combine_left(intervals, min_i)
        else:
            conn_l = connectivity(captcha, intervals[min_i-1][1]+1, intervals[min_i][0]-1)
            conn_r = connectivity(captcha, intervals[min_i][1]+1, intervals[min_i+1][0]-1)
            if conn_l and conn_r:
                intervals = combine_right(intervals, min_i)
            elif conn_l and not conn_r:
                intervals = combine_left(intervals, min_i)
            elif not conn_l and conn_r:
                intervals = combine_right(intervals, min_i)
            else:
                if_l = combine_left(intervals, min_i)[min_i-1]
                if_r = combine_right(intervals, min_i)[min_i]
                if_l_len = if_l[1] - if_l[0]
                if_r_len = if_r[1] - if_r[0]
                if if_l_len < if_r_len:
                    intervals = combine_left(intervals, min_i)
                elif if_r_len < if_l_len:
                    intervals = combine_right(intervals, min_i)
                else:
                    intervals = combine_left(intervals, min_i)
    if len(intervals) < 4:
        in_len = numpy.array([inte[1]-inte[0] for inte in intervals])
        big_idx = numpy.argmax(in_len)
        biggest = intervals[big_idx]
        big_len = biggest[1] - biggest[0]
        split_no = 5 - len(intervals)
        split_points = [biggest[0]+big_len*i/split_no for i in range(split_no+1)]
        splitted = []
        for i in range(len(split_points)-1):
            splitted.append((split_points[i], split_points[i+1]))
        intervals = intervals[:big_idx] + splitted + intervals[(big_idx+1):]
    return intervals

def split_y(captcha):
    """this should be invoked after `split_pic`"""
    y_up = 0
    y_down = 50
    for i in range(len(captcha)):
        if captcha[i, :].sum() >= 255*3:
            y_up = i
            break
    for i in range(len(captcha)-1, 0, -1):
        if captcha[i, :].sum() >= 255*3:
            y_down = i+1
            break
    return y_up, y_down

def combine_left(intervals, i):
    return intervals[:(i-1)] + [(intervals[i-1][0], intervals[i][1])] + intervals[(i+1):]

def combine_right(intervals, i):
    return intervals[:i] + [(intervals[i][0], intervals[i+1][1])] + intervals[(i+2):]

def connectivity(captcha, y1, y2):
    for y in range(y1, y2+1):
        if captcha[:, y].sum() < 255:
            return False
    return True

def del_line(captcha, line_no=2):
    x1s = []
    x2s = []
    for i in range(0, 50):
        if captcha[i, 0] > 0:
            x1s.append(i)
        if captcha[i, 99] > 0:
            x2s.append(i)
    lines = [(x1, x2) for x1 in x1s for x2 in x2s]
    result = captcha
    for i in range(line_no):
        result = optimal_result(result, lines)
    return del_dot(result)

def regularize(captcha):
    """regularize to 40*40"""
    x, y = captcha.shape
    if x > 40:
        to_del = x - 40
        del_u = to_del/2
        del_l = to_del - del_u
        captcha = captcha[del_u:(-del_l), :]
    elif x < 40:
        to_pad = 40 - x
        pad_u = to_pad/2
        pad_l = to_pad - pad_u
        captcha = numpy.vstack((numpy.zeros((pad_u, y), dtype=int), captcha, numpy.zeros((pad_l, y), dtype=int)))
    if y > 40:
        to_del = y - 40
        del_l = to_del/2
        del_r = to_del - del_l
        captcha = captcha[:, del_l:(-del_r)]
    elif y < 40:
        to_pad = 40 - y
        pad_l = to_pad/2
        pad_r = to_pad - pad_l
        captcha = numpy.c_[numpy.zeros((40, pad_l), dtype=int), captcha, numpy.zeros((40, pad_r), dtype=int)]
    return captcha

def explorer():
    for c in range(0, len(captchas), 77):
        e = del_line(captchas[c])
        pl.figure(c)
        for i, p in enumerate(split_pic(e)):
            pl.subplot(221+i)
            char = e[:, p[0]:p[1]]
            y1, y2 = split_y(char)
            pl.imshow(regularize(char[y1:y2, :]), cmap=pl.cm.Greys)
        pl.show()
        if raw_input() == 'q':
            pl.close('all')
            break

def proc_split(start, end, captchas):
    chars = []
    for i, c in enumerate(captchas):
        e = del_line(c)
        for p in split_pic(e):
            char = e[:, p[0]:p[1]]
            y1, y2 = split_y(char)
            chars.append(numpy.reshape(regularize(char[y1:y2, :]), (-1,)))
    return chars

if __name__ == '__main__':
    import pp
    ppservers = ()
    job_server = pp.Server(ppservers=ppservers)
    step = 100
    inputs = [(i, i+step, captchas[i:(i+step)]) for i in range(0, 10000, step)]
    jobs = [job_server.submit(proc_split, inp, (del_line, del_dot, split_pic, get_pic, optimal_result, combine_left, combine_right, connectivity, split_y, regularize), ('numpy',)) for inp in inputs]
    import itertools
    chars = itertools.chain.from_iterable([job() for job in jobs])
    numpy.save('clean_chars', numpy.array(list(chars)))
    job_server.print_stats()
