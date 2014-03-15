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

def optimal_result(captcha):
    x1s = []
    x2s = []
    for i in range(0, 50):
        if captcha[i, 0] > 0:
            x1s.append(i)
        if captcha[i, 99] > 0:
            x2s.append(i)
    lines = [(x1, x2) for x1 in x1s for x2 in x2s]
    min_area = 5250
    best_result = None
    for line in lines:
        t = captcha - get_pic(*line)
        t = numpy.where(t>0, 1, 0)
        t_area = t.sum()
        if t_area < min_area:
            min_area = t_area
            best_result = t
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
    return intervals

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
    result = captcha
    for i in range(line_no):
        result = optimal_result(result)
    return del_dot(result)

def to_flat(captcha):
    return numpy.insert(numpy.reshape(captcha, (-1,)), 0, captcha.shape[1])

def explorer():
    for c in range(0, len(captchas), 77):
        e = del_line(captchas[c])
        pl.figure(c)
        for i, p in enumerate(split_pic(e)):
            pl.subplot(141+i)
            pl.imshow(e[:, p[0]:p[1]], cmap=pl.cm.Greys)
        pl.show()
        if raw_input() == 'q':
            pl.close('all')
            break

def proc_split(start, end, captchas):
    chars = []
    for c in captchas:
        e = del_line(c)
        partitions = split_pic(e)
        if len(partitions) == 4:
            for p in split_pic(e):
                chars.append(to_flat(e[:, p[0]:p[1]]))
    numpy.save('splitted_chars%d-%d.npy' % (start, end-1), numpy.array(chars))

if __name__ == '__main__':
    import pp
    ppservers = ()
    job_server = pp.Server(ppservers=ppservers)
    step = 100
    inputs = [(i, i+step, captchas[i:(i+step)]) for i in range(0, 10000, step)]
    jobs = [job_server.submit(proc_split, inp, (del_line, del_dot, split_pic, to_flat, get_pic, optimal_line, combine_left, combine_right, connectivity,), ('numpy',)) for inp in inputs]
    _ = [job() for job in jobs]
    job_server.print_stats()
