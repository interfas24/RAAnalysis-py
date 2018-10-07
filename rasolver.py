from hornpattern import PyramidalHorn, get_default_pyramidal_horn
from arrayinfo import RAInfo
from sources import PlaneWave, Source
from rautils import gsinc, ideal_ref_unit, dB
import numpy as np
import matplotlib.pyplot as plt

"""
1. array info (cell size scale | horns)
2. get phase
3. get tetm
4. add task
5. calc
"""

class RASolver:
    R = 100.

    def __init__(self, rainfo):
        self.rainfo = rainfo

    def __erxy_fft(self, u, v):
        px, py = self.rainfo.get_pxy()
        Nx, Ny = self.rainfo.get_Nxy()
        dx, dy = self.rainfo.get_dxy()
        k0 = self.rainfo.get_k0()
        K1 = np.exp(-1j*k0/2.0 * (u*(Nx-1)*dx + v*(Ny-1)*dy))

        d_sumx, d_sumy = 0.0, 0.0
        for (i, (Exmn, Eymn)) in list(enumerate(self.rainfo)):
            n, m = int(i / Nx), int(i % Nx)
            ejk = np.exp(1j*k0*(u*m*dx+v*n*dy))
            d_sumx += (Exmn * ejk)
            d_sumy += (Eymn * ejk)

        A = K1 * px *py * gsinc(k0*u*px/2.0) * gsinc(k0*v*py/2.0)
        return A*d_sumx, A*d_sumy

    def __calc_one_point(self, r, t, p):
        u = np.sin(t) * np.cos(p)
        v = np.sin(t) * np.sin(p)

        Erx, Ery = self.__erxy_fft(u, v)

        k0 = self.rainfo.get_k0()
        R = r
        E_phi = -1j*k0*np.exp(-1j*k0*R)/(2*np.pi*R)*np.cos(t) * (Erx * np.sin(p) - Ery * np.cos(p))
        E_theta = 1j*k0*np.exp(-1j*k0*R)/(2*np.pi*R) * (Erx * np.cos(p) + Ery * np.sin(p))
        E_total = np.sqrt(E_phi*E_phi + E_theta*E_theta)

        return np.abs(E_total)*np.abs(E_total)*R*R / (2 * 377.)


    def append_task(self):
        pass

    def run(self):

        pass

    def test(self):
        N = 300
        ts = np.linspace(np.deg2rad(-89), np.deg2rad(89), N)
        ps = np.deg2rad(0)

        res = []
        for t in ts:
            #print(np.rad2deg(t))
            res.append(self.__calc_one_point(100., t, ps))
        res = dB(res)
        plt.figure()
        plt.plot(np.rad2deg(ts), res)
        plt.show()


def test1():
    freq = 5e9
    cell_sz = 30. / 1000.
    scale = 20

    abg = (np.deg2rad(180), np.deg2rad(180), np.deg2rad(0))
    src = Source()
    src.append(get_default_pyramidal_horn(freq), abg, (0., 0., 0.5))
    tp = [(np.deg2rad(30), np.deg2rad(0))]
    tpm = [(np.deg2rad(0), np.deg2rad(0), 1)]
    foci = [(0, 0, 0.8, 1.0)]

    arr = RAInfo(src, cell_sz, (scale, scale), ('pencil', tp), ideal_ref_unit)
    solver = RASolver(arr)
    solver.test()


if __name__ == '__main__':
    test1()
