from hornpattern import PyramidalHorn, get_default_pyramidal_horn
from rautils import sol, make_v_mtx, car2sph, sph2car, car2sph_mtx, sph2car_mtx, create_array_pos
import numpy as np
import matplotlib.pyplot as plt


class PlaneWave:
    def __init__(self, freq):
        k0 = 2 * np.pi / (sol / freq)
        self.e0 = 200.0
        self.k0 = k0
        self.freq = freq

    def frequency(self):
        return self.freq

    def get_k0(self):
        return self.k0

    def have_input_power(self):
        return False

    def efield_at_xyz(self, x, y, z):
        ex = 0j
        ey = self.e0 * np.exp(1j*self.k0*z)
        ez = 0j
        Exyz = make_v_mtx(ex, ey, ez)
        r, t, p = car2sph(x, y, z)
        Ertp = car2sph_mtx(t, p) * Exyz
        return Exyz, Ertp.item(1), Ertp.item(2)

    def efield_at_rtp(self, r, t, p):
        x, y, z = sph2car(r, t, p)
        return self.efield_at_xyz(x, y, z)


class Source:
    def __init__(self):
        self.src = []
        self.abg = []
        self.pos = []
        self.idx = 0
        self.type = ''

    def __len__(self):
        return len(self.src)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == len(self):
            raise StopIteration
        ret = self.src[self.idx], self.abg[self.idx], self.pos[self.idx]
        self.idx += 1
        return ret

    def is_horn(self):
        return isinstance(self.src[0], PyramidalHorn)

    def frequency(self):
        if len(self) < 1:
            raise ValueError
        return self.src[0].frequency()

    def k0(self):
        if len(self) < 1:
            raise ValueError
        return self.src[0].get_k0()

    def append(self, src, abg, pos):
        if self.type and self.type != type(src):
            print('source should be in same type')
            raise TypeError
        self.src.append(src)
        self.abg.append(abg)
        self.pos.append(pos)
        self.type = type(src)

def test_plane_wave():
    f = 5.0e9
    cell_sz = 15. / 1000.
    scale = 50
    z = 10.

    pw = PlaneWave(f)
    xl, yl = create_array_pos(cell_sz, scale, scale, ex=True)
    magE = np.ndarray((len(yl), len(xl)))
    pE = np.ndarray((len(yl), len(xl)))
    for (yi, y) in list(enumerate(yl)):
        for (xi, x) in list(enumerate(xl)):
            Exyz, _, _ = pw.efield_at_xyz(x, y, z)
            mag = np.sqrt(Exyz.item(0)**2 + Exyz.item(1)**2 + Exyz.item(2)**2)
            pha = np.angle(Exyz.item(1))
            print(mag, np.rad2deg(pha))
            #magE[yi, xi] = np.abs(mag)
            magE[yi, xi] = mag.real
            pE[yi, xi] = pha

    plt.figure()
    plt.pcolor(xl, yl, magE)

    plt.figure()
    plt.pcolor(xl, yl, pE)
    plt.show()


if __name__ == '__main__':
    srcs = Source()
    f = 10e9
    srcs.append(get_default_pyramidal_horn(f), (np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)), (0, 0, 1.0))
    srcs.append(get_default_pyramidal_horn(f), (np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)), (1.0, 1.0, 1.0))
    #srcs.append(PlaneWave(f), (np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)), (1.0, 1.0, 1.0))

    for s, abg, pos in srcs:
        print(abg, pos)
