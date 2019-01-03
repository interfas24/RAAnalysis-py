import numpy as np
import matplotlib.pyplot as plt
from rautils import dB, distance, sol, create_array_pos, car2sph, waveimpd
from scipy import integrate


class PyramidalHorn:
    def __init__(self, r1, r2, a, b, a1, b1, E0, freq, initp=0.0):
        k0 = 2 * np.pi / (sol / freq)

        self.r1 = r1
        self.r2 = r2
        self.a = a
        self.b = b
        self.a1 = a1
        self.b1 = b1
        self.E0 = E0
        self.k0 = k0
        self.rz = 1
        self.freq = freq
        self.initp = initp

    def have_input_power(self):
        return True

    def frequency(self):
        return self.freq

    def get_k0(self):
        return self.k0

    def integ_func(self, t, p):
        R = 100.    # any R would be ok
        _, et, ep = self.efield_at_rtp(R, t, p)
        etp = np.sqrt(et**2 + ep**2)
        return np.abs(etp)**2 *R*R*np.sin(t) / (2*waveimpd)

    def efield_at_xyz(self, x, y, z):
        r, theta, phi = car2sph(x, y, z)
        return self.efield_at_rtp(r, theta, phi)


    def efield_at_rtp(self, r, theta, phi):
        k = 2*np.pi
        ky = k * np.sin(theta)*np.sin(phi)
        t1 = np.sqrt(1/(np.pi*k*self.r1)) * (-k*self.b1/2-ky*self.r1)
        t2 = np.sqrt(1/(np.pi*k*self.r1)) * (k*self.b1/2-ky*self.r1)

        kxp = k*np.sin(theta)*np.cos(phi) + np.pi/self.a1
        kxdp = k*np.sin(theta)*np.cos(phi) - np.pi/self.a1
        t1p = np.sqrt(1/(np.pi*k*self.r2)) * (-k*self.a1/2-kxp*self.r2)
        t2p = np.sqrt(1/(np.pi*k*self.r2)) * (k*self.a1/2-kxp*self.r2)

        t1dp = np.sqrt(1/(np.pi*k*self.r2)) * (-k*self.a1/2-kxdp*self.r2)
        t2dp = np.sqrt(1/(np.pi*k*self.r2)) * (k*self.a1/2-kxdp*self.r2)

        I1 = .5*np.sqrt(np.pi*self.r2/k) * (np.exp(1j*kxp**2*self.r2/(2*k)) * (self.Fresnel(t2p)-self.Fresnel(t1p))
                                            + np.exp(1j*kxdp**2*self.r2/(2*k)) * (self.Fresnel(t2dp) - self.Fresnel(t1dp)))
        I2 = np.sqrt(np.pi*self.r1/k) * np.exp(1j*ky**2*self.r1/(2*k)) * (self.Fresnel(t2) - self.Fresnel(t1))

        k = self.k0
        Etheta = 1j*k*self.E0*np.exp(-1j*(k*r + self.initp))/(4*np.pi*r) * (np.sin(phi) * (1+np.cos(theta)) * I1 * I2)
        Ephi = 1j*k*self.E0*np.exp(-1j*(k*r + self.initp))/(4*np.pi*r) * (np.cos(phi) * (1+np.cos(theta)) * I1 * I2)

        tmtx = np.matrix(
            [
                [np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)],
                [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi), np.cos(phi)],
                [np.cos(theta), -np.sin(theta), 0]
            ]
        )

        emtx = np.matrix([
            [0],
            [Etheta],
            [Ephi]
        ])
        return tmtx * emtx, Etheta, Ephi


    def Fresnel(self, x):
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


def get_horn_input_power(horn):
    if not isinstance(horn, PyramidalHorn):
        raise ValueError
    ret = integrate.nquad(horn.integ_func, [[0, np.pi/2.], [0, np.pi*2]])
    return ret[0]

def get_default_pyramidal_horn(freq, E0=10.0, initp=0.0):
    return PyramidalHorn(3.56, 5.08, 0.762, 0.3386, 1.524, 1.1854, E0, freq, initp)


def test_horn():
    f = 5.0e9
    cell_sz = 15. / 1000.
    scale = 50
    z = cell_sz*scale*1

    phorn = PyramidalHorn(3.56, 5.08, 0.762, 0.3386, 1.524, 1.1854, 10, f, initp=np.deg2rad(180))
    xl, yl = create_array_pos(cell_sz, scale, scale, ex=True)
    magE = np.ndarray((len(yl), len(xl)))
    pE = np.ndarray((len(yl), len(xl)))
    for (yi, y) in list(enumerate(yl)):
        for (xi, x) in list(enumerate(xl)):
            Exyz, _, _ = phorn.efield_at_xyz(x, y, z)
            mag = np.sqrt(Exyz.item(0)**2 + Exyz.item(1)**2 + Exyz.item(2)**2)
            pha = np.angle(Exyz.item(1))
            magE[yi, xi] = np.abs(mag)
            pE[yi, xi] = pha

    plt.figure()
    plt.pcolor(xl, yl, magE, cmap='jet')

    plt.figure()
    plt.pcolor(xl, yl, pE, cmap='jet')

    plt.show()

if __name__ == '__main__':
    test_horn()
