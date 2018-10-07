import numpy as np
from rautils import sol, distance, create_array_pos
import matplotlib.pyplot as plt


class BeamForming:
    def __init__(self, freq, xpos, ypos, fpos):
        """
        Parameters:
            freq    unit:Hz
            xpos    x-axis element position list unit:m
            ypos    y-axis element position list unit:m
            fpos    feed position tuple, list etc   unit:m
        """
        self.freq = freq
        self.k0 = 2 * np.pi / (sol / freq)
        self.xpos = xpos
        self.ypos = ypos
        self.fpos = fpos

    def frequency(self):
        return self.freq

    def __get_phi_n(self, theta, phi, x, y):
        return -self.k0 * np.sin(theta) * np.cos(phi) * x - self.k0 * np.sin(theta) * np.sin(phi) * y


    def __feed_phase(self, x, y):
        sum = 0.0j
        for (fx, fy, fz) in self.fpos:
            df = distance((fx, fy, fz), (x, y, 0.0))
            sum += np.exp(1j*df*self.k0)
        return np.angle(sum) + np.pi*2


    def __get_phi_k(self, theta, phi, x, y):
        return np.arctan2(-np.cos(theta)*np.sin(phi)*x + np.cos(theta)*np.cos(phi)*y, np.cos(phi)*x + np.sin(phi)*y)


    def form_pencil_beam(self, tp, haskd=True):
        """
        :param tp: theta phi [(t1, p1), (t1, p1), ...], unit rad
        :param haskd: use kd with fpos
        :return: np.ndarray unit rad
        """
        ret = np.ndarray((len(self.ypos), len(self.xpos)))
        for (yidx, y) in list(enumerate(self.ypos)):
            for (xidx, x) in list(enumerate(self.xpos)):
                sum = 0.0j
                for(t, p) in tp:
                    phi_n = self.__get_phi_n(t, p, x, y)
                    sum += np.exp(1j*phi_n)
                if haskd:
                    #ret[yidx][xidx] = (np.angle(sum) + self.k0*distance(self.fpos, (x, y, 0.0))) % (2*np.pi)
                    ret[yidx][xidx] = (np.angle(sum) + self.__feed_phase(x, y)) % (2*np.pi)
                else:
                    ret[yidx][xidx] = (np.angle(sum) + 2*np.pi) % (2*np.pi)
        return ret


    def form_focal_beam(self, focals, haskd=True):
        """
        :param focals: [(dx1, dy1, dz1, D1), (dx2, dy2, dz2, D2), ...]
        :param haskd:
        :return:
        """
        ret = np.ndarray((len(self.ypos), len(self.xpos)))
        for (yidx, y) in list(enumerate(self.ypos)):
            for (xidx, x) in list(enumerate(self.xpos)):
                sum = 0.0j
                for (dx, dy, dz, D) in focals:
                    sum += D * np.exp(1j*self.k0*distance((dx, dy, dz), (x, y, 0.0)))
                if haskd:
                    ret[yidx][xidx] = (np.angle(sum) + self.__feed_phase(x, y)) % (2*np.pi)
                else:
                    ret[yidx][xidx] = (np.angle(sum) + 2*np.pi) % (2*np.pi)
        return ret

    def form_oam_beam(self, tpm, alpha=0.0, haskd=True):
        """
        :param tpm:
        :param haskd:
        :param focal:
        :return:
        """
        ret = np.ndarray((len(self.ypos), len(self.xpos)))
        for (yidx, y) in list(enumerate(self.ypos)):
            for (xidx, x) in list(enumerate(self.xpos)):
                sum = 0.0j
                for (t, p, mode) in tpm:
                    pn = self.__get_phi_n(t, p, x, y)
                    pk = self.__get_phi_k(t, p, x, y)
                    big_phi = pn + pk
                    sum += np.exp(1j*(big_phi))
                if haskd:
                    ret[yidx][xidx] = (np.angle(sum) + self.__feed_phase(x, y) + np.sqrt(x*x+y*y)*np.sin(alpha)*self.k0)\
                                      % (2*np.pi)
                else:
                    ret[yidx][xidx] = (np.angle(sum) + 2*np.pi + np.sqrt(x*x+y*y)*np.sin(alpha)*self.k0) % (2*np.pi)
        return ret


if __name__ == '__main__':
    f = 10.0e9
    cell_sz = 15.0 / 1000.
    scale = 20
    fdr = 0.8
    oa = np.deg2rad(0)
    fp1 = (0., -cell_sz*scale*fdr*np.sin(oa), cell_sz*scale*fdr*np.cos(oa))
    fp2 = (0., cell_sz*scale*fdr*np.sin(oa), cell_sz*scale*fdr*np.cos(oa))
    fpos = [fp1]
    xlex, ylex = create_array_pos(cell_sz, scale, scale, True)
    bf = BeamForming(f, xlex, ylex, fpos)

    beam_dir = [(np.deg2rad(0), np.deg2rad(0))]
    pb = bf.form_pencil_beam(beam_dir, True)
    plt.figure()
    plt.pcolor(xlex, ylex, pb)

    focals = [(0.0, 0.0, 0.5, 1)]
    fb = bf.form_focal_beam(focals)
    plt.figure()
    plt.pcolor(xlex, ylex, fb)

    tpm = [(np.deg2rad(0), np.deg2rad(0), 1)]
    ob = bf.form_oam_beam(tpm, alpha=np.deg2rad(10))
    plt.figure()
    plt.pcolor(xlex, ylex, ob)
    plt.show()










