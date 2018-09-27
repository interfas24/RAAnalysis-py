import numpy as np


sol = 299792458   # speed of light    unit:m/s

def distance(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)

def create_array_pos(cell_sz, scalex, scaley, ex):
    nx = scalex+1 if ex else scalex
    ny = scaley+1 if ex else scaley
    xlist = np.linspace(-cell_sz*scalex/2.+cell_sz/2., cell_sz*scalex/2+cell_sz/2., nx, endpoint=ex)
    ylist = np.linspace(-cell_sz*scaley/2.+cell_sz/2., cell_sz*scaley/2+cell_sz/2., ny, endpoint=ex)
    return xlist, ylist

def gsinc(x, k=1.0):
    return 1.0 if x == 0.0 else np.sin(k*x) / (k*x)

def dB(dat, type=''):
    ret = dat.copy()
    for (i, d) in list(enumerate(ret)):
        factor = 20. if type == 'power' else 10.0
        ret[i] = factor * np.log10(d)
    return ret

def gain2mag(gain):
    ret = gain.copy()
    for (i, g) in list(enumerate(ret)):
        ret[i] = 10. ** (g / 10.)
    return ret

def norm2zero(dat):
    md = np.max(dat)
    ret = dat.copy()
    for (i, d) in list(enumerate(ret)):
        ret[i] = d - md
    return ret


if __name__ == '__main__':
    print(distance((1,2,3), (4,5,6)), np.sqrt(27))
    cell_sz = 10.
    scale = 10.
    print(create_array_pos(cell_sz, scale, scale, False))
    print(create_array_pos(cell_sz, scale, scale, True))
    print(gsinc(10), np.sin(10)/10.)
    print(gsinc(0))
    print(dB([1,2,3,4]), gain2mag(dB([1,2,3,4])))
    print(dB([1,2,3,4], 'power'))
    print(norm2zero([-2, -3, 1, 2, 4, -7]))
