import numpy as np
from rautils import farR, sph2car_mtx, make_v_mtx, dB
import threading

import matplotlib.pyplot as plt

class EfieldResult:
    def __init__(self, Etheta, Ephi, pos):
        (r, t, p) = pos
        self.Etheta = Etheta
        self.Ephi = Ephi
        self.Pos = pos
        self.Etotal = np.sqrt(Etheta*Etheta + Ephi*Ephi)
        xyz_mtx = sph2car_mtx(t, p) * make_v_mtx(0j, Etheta, Ephi)
        self.Ex = xyz_mtx.item(0)
        self.Ey = xyz_mtx.item(1)
        self.Ez = xyz_mtx.item(2)

    def get_etotal(self):
        return self.Etotal

    def get_sph_field(self):
        return 0j, self.Etheta, self.Ephi

    def get_car_field(self):
        return self.Ex, self.Ey, self.Ez

    def get_directivity(self):
        (r, t, p) = self.Pos
        return np.abs(self.Etotal)*np.abs(self.Etotal)*r*r/(2*377.)

    def get_gain(self, ipower):
        return 4*np.pi*np.abs(self.Etotal)*np.abs(self.Etotal) * farR * farR / ipower

class TaskState:
    Pending = 1
    Calculating = 2
    Finished = 3

class Task:
    def __init__(self, oid, pos):
        self.oid = oid
        self.pos = pos
        self.results = np.empty(len(pos), dtype=EfieldResult)
        self.cnt = 0
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.pos)

    def __iter__(self):
        with self.lock:
            self.cnt = 0
            return self

    def __next__(self):
        with self.lock:
            if self.cnt == len(self):
                raise StopIteration
            ret = self.pos[self.cnt]
            self.cnt += 1
            return ret

    def set_current_result(self, efield):
        with self.lock:
            self.results[self.cnt-1] = efield

    def get_old_idx(self):
        return self.oid

    def get_results(self):
        return self.results


class FarZone:
    def __init__(self, row, col):
        self.state = TaskState.Pending
        self.row = row
        self.col = col
        self.nrow = len(row)
        self.ncol = len(col)
        self.alldat = np.empty(self.nrow*self.ncol, dtype=EfieldResult)
        self.lock = threading.Lock()

    def __len__(self):
        return self.nrow * self.ncol

    """
    def __setitem__(self, key, value):
        with self.lock:
            if key > len(self):
                print('index exceeds')
                raise ValueError
            self.alldat[key] = value
    """

    """
    def __getitem__(self, item):
        return self.alldat[item]
    """

    def set_results(self, tsk):
        with self.lock:
            b, e = tsk.get_old_idx()
            for i in range(b, e):
                self.alldat.put(i, tsk.get_results()[i-b])

    def assign_task(self, mp=200):
        if len(self) <= mp:
            pos = [(farR, t, p) for t in self.col for p in self.row]
            return [Task((0, len(self)), pos)]
        else:
            tsk = []
            b, e = 0, 0
            pos = []
            for p in self.row:
                for t in self.col:
                    pos.append((farR, t, p))
                    e += 1
                    if e - b == mp:
                        tsk.append(Task((b, e), pos))
                        pos = []
                        b = e
            tsk.append(Task((b, e), pos))
            return tsk


class Gain2D(FarZone):
    def __init__(self, phi, ntheta):
        ts = np.linspace(-np.pi/2, np.pi/2, ntheta)
        super().__init__([phi], ts)

    def post_process(self, integ, plot=False):
        fields = np.reshape(self.alldat, (self.nrow, self.ncol))
        gs = [x.get_gain(integ) for x in fields[0]]
        gs = dB(gs)

        if plot:
            plt.figure()
            plt.plot(self.col, gs)
            plt.show()
        return [self.col, gs]


class Gain3D(FarZone):
    def __init__(self, nphi, ntheta):
        ps = np.linspace(0, np.pi*2, nphi)
        ts = np.linspace(-np.pi/2, np.pi/2, ntheta)
        super().__init__(ps, ts)

    def post_process(self):
        fields = np.reshape(self.alldat, (self.nrow, self.ncol))
        #print(fields)
        id = int(self.nrow / 2)
        integ = 22732.769823328235
        gs = [x.get_gain(integ) for x in fields[10]]
        gs = dB(gs)

        plt.figure()
        plt.plot(self.col, gs)
        plt.show()


class Directivity2D(FarZone):
    pass

class Directivity3D(FarZone):
    pass

class FresnelPlane:
    pass


if __name__ == '__main__':
    g2d = Gain2D(np.deg2rad(0), 100.)
    tsks = g2d.assign_task()

    for (i, tsk) in list(enumerate(tsks)):
        print('in task {}'.format(i))
        for (r, t, p) in tsk:
            print('r={},t={},p={}'.format(r, t, p))
        print('\n')


