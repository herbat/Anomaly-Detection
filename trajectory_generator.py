import time
import math
import pickle
import os, sys
import lightnet
import itertools
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

base_path = 'video_data/1/Train/Train001/'


def get_jpg(fname):
    _size = ()
    im = None
    outfile = base_path + '_tmp.jpg'
    try:
        print(fname)
        im = Image.open(os.path.join(base_path, fname))
        _size = im.size
        im.thumbnail(im.size)
        im.save(outfile, 'JPEG', quality=100)
    except FileNotFoundError:
        print('File not found :/')
    except Exception:
        print('Something pretty bad happened :O')

    return lightnet.Image.from_bytes(open(os.path.join(base_path, '_tmp.jpg'), 'rb').read()), _size, im


# this function relates the new objects to the previous
def track(oldarr, newarr, _size, _collect):

    max_t = 0 if _collect.keys().__len__() == 0 else max(_collect.keys())

    for i in newarr:
        _min = -1
        bound = itertools.product((0, 0), _size)
        # find the minimum distance from old items
        for j in oldarr:
            dist = math.sqrt((j[1]-i[1])**2 + (j[2]-i[2])**2)
            if _min == -1 or dist < _min:
                _min = dist
                i[0] = j[0]

        # if the new item is closer to the boundary than every old item, consider it a new item
        for j in bound:
            dist = math.sqrt((j[0] - i[1]) ** 2 + (j[1] - i[2]) ** 2)
            if _min == -1 or dist < _min:
                _min = dist
                i[0] = max_t + 1

        if i[0] in _collect:
            _collect[i[0]].append([i[1], i[2]])
        else:
            _collect[i[0]] = [[i[1], i[2]]]

    return _collect


def plot_trajectory(_c, im):
    for t in _c.keys():
        x = []
        y = []
        for i, j in _c[t]:
            x.append(i)
            y.append(j)
        plt.plot(x, y, zorder=1)

    plt.imshow(im, zorder=0)
    plt.axis([0, 238, 158, 0])
    plt.show()


model = lightnet.load('yolo')

start = time.time()

collect = {}

# for dir in os.scandir(base_path):
prev_arr = []
for i in range(200):
    fname = str(i+1).zfill(3) + '.tif'
    image, size, im_p = get_jpg(fname)
    tmp = model(image)
    result = []

    for i, e in enumerate(tmp):
        result.append([i, e[-1][0], e[-1][1]])

    if not prev_arr == []:
        collect = track(prev_arr, result, size, collect)
        plot_trajectory(collect, im_p)

    prev_arr = result

end = time.time()

pickle.dump(collect, open('trajectory_data/tr001_1.p', 'wb'))
print(end-start)