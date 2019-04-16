import numpy as np


sol = 299792458   # speed of light    unit:m/s
farR = 10
waveimpd = 119.9169832 * np.pi

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

def sph2car_mtx(theta, phi):
    return np.matrix(
                [
                    [np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)],
                    [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi), np.cos(phi)],
                    [np.cos(theta), -np.sin(theta), 0]
                ]
            )

def car2sph_mtx(theta, phi):
    return sph2car_mtx(theta, phi).transpose()

def sph2car(r, theta, phi):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x, y, z

def car2sph(x, y, z):
    r = np.sqrt(x*x+y*y+z*z)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    return r, theta, phi

def make_v_mtx(a, b, c):
    return np.matrix(
        [
            [a],
            [b],
            [c]
        ]
    )

def R2F(alpha, beta, gamma):
    mtx1 = np.matrix([[np.cos(gamma), np.sin(gamma), 0],
                      [-np.sin(gamma), np.cos(gamma), 0],
                      [0, 0, 1]])
    mtx2 = np.matrix([[1, 0, 0],
                      [0, np.cos(beta), np.sin(beta)],
                      [0, -np.sin(beta), np.cos(beta)]])
    mtx3 = np.matrix([[np.cos(alpha), np.sin(alpha), 0],
                      [-np.sin(alpha), np.cos(alpha), 0],
                      [0, 0, 1]])

    A_cc = mtx1 * mtx2 * mtx3
    return A_cc

def F2R(alpha, beta, gamma):
    return R2F(alpha, beta, gamma).transpose()

def ideal_ref_unit(pha, amp=1.0, bits=None, pol='Y'):
    ret = []
    for i in range(len(pha)):
        sp = pha[i]
        if bits != None:
            step = np.pi*2/(2**bits)
            sp = int(pha[i]/step) * step

        if pol == 'X':
            sparam = (amp*np.exp(1j*sp), 0j, 0j, 1+0j)
        elif pol == 'Y':
            sparam = (1+0j, 0j, 0j, amp*np.exp(1j*sp))
        elif pol == 'XY':
            sparam = (amp*np.exp(1j*sp), 0j, 0j, amp*np.exp(1j*sp))
        else:
            sparam = (amp*np.exp(1j*sp), 0j, 0j, amp*np.exp(1j*(sp-np.pi/2.0)))

        ret.append(sparam)

    return ret

def far_field_distance(freq, maxlen):
    """
    Calculate far-field distance
    :param freq: unit : GHz
    :param maxlen: max length of the aperture antenna, unit : m
    :return: far-field distance, unit : m
    """
    return 2 * maxlen * maxlen / (sol / freq)


def Fresnel(x):
    A = [
        1.595769140,
        -0.000001702,
        -6.808508854,
        -0.000576361,
        6.920691902,
        -0.016898657,
        -3.050485660,
        -0.075752419,
        0.850663781,
        -0.025639041,
        -0.150230960,
        0.034404779
    ]

    B = [
        -0.000000033,
        4.255387524,
        -0.000092810,
        -7.780020400,
        -0.009520895,
        5.075161298,
        -0.138341947,
        -1.363729124,
        -0.403349276,
        0.702222016,
        -0.216195929,
        0.019547031
    ]

    C = [
        0,
        -0.024933975,
        0.000003936,
        0.005770956,
        0.000689892,
        -0.009497136,
        0.011948809,
        -0.006748873,
        0.000246420,
        0.002102967,
        -0.001217930,
        0.000233939
    ]

    D = [
        0.199471140,
        0.000000023,
        -0.009351341,
        0.000023006,
        0.004851466,
        0.001903218,
        -0.017122914,
        0.029064067,
        -0.027928955,
        0.016497308,
        -0.005598515,
        0.000838386
    ]

    if x == 0:
        return 0
    elif x < 0:
        x = np.abs(x)
        x = (np.pi/2) * (x**2)
        F = 0
        if x < 4:
            for k in range(12):
                F += (A[k] + 1j*B[k]) * ((x/4) ** k)
            return -(F*np.sqrt(x/4) * np.exp(-1j*x))
        else:
            for k in range(12):
                F += (C[k] + 1j*D[k]) * ((4/x) ** k)
            return -(F*np.sqrt(4/x) * np.exp(-1j*x) + (1-1j)/2)
    else:
        x = (np.pi/2) * (x**2)
        F = 0
        if x < 4:
            for k in range(12):
                F += (A[k] + 1j*B[k]) * ((x/4) ** k)
            return F*np.sqrt(x/4) * np.exp(-1j*x)
        else:
            for k in range(12):
                F += (C[k] + 1j*D[k]) * ((4/x) ** k)
            return F*np.sqrt(4/x) * np.exp(-1j*x) + (1-1j)/2


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

    xyz = [1, 2, 3]
    r, t, p = car2sph(*xyz)
    xyz_mtx = make_v_mtx(*xyz)
    print(car2sph_mtx(t, p)*xyz_mtx)
    print(sph2car_mtx(t, p)*car2sph_mtx(t, p)*xyz_mtx)
    print(waveimpd)
