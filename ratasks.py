import numpy as np
from rautils import farR, sph2car_mtx, make_v_mtx, dB, car2sph
import threading
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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
        self.R = farR

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

    def set_R(self, r):
        self.R = r

    def set_results(self, tsk):
        with self.lock:
            b, e = tsk.get_old_idx()
            for i in range(b, e):
                self.alldat.put(i, tsk.get_results()[i-b])

    def assign_task(self, mp=200):
        if len(self) <= mp:
            pos = [(self.R, t, p) for t in self.col for p in self.row]
            return [Task((0, len(self)), pos)]
        else:
            tsk = []
            b, e = 0, 0
            pos = []
            for p in self.row:
                for t in self.col:
                    pos.append((self.R, t, p))
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

    def post_process(self, integ, fig=False):
        fields = np.reshape(self.alldat, (self.nrow, self.ncol))
        gs = [x.get_gain(integ) for x in fields[0]]
        gs = dB(gs)

        if fig:
            plt.figure()
            plt.plot(self.col, gs)
            plt.show()
        return [self.col, gs]


class Gain3D(FarZone):
    def __init__(self, nphi, ntheta):
        self.ps = np.linspace(0, np.pi*2, nphi)
        self.ts = np.linspace(-np.pi/2, np.pi/2, ntheta)
        super().__init__(self.ps, self.ts)

    def post_process(self, integ, fig=False):
        fields = np.reshape(self.alldat, (self.nrow, self.ncol))
        gs = np.ndarray(shape=fields.shape)

        if fig:
            for (id, field) in list(enumerate(fields)):
                g = dB([x.get_gain(integ) for x in field])
                gs[id,:] = np.array(g)

            T, P = np.meshgrid(self.ts, self.ps)

            for i in range(len(self.ps)):
                for j in range(len(self.ts)):
                    if gs[i,j] < -10:
                        gs[i, j] = -10

            gs = gs - np.min(gs)

            X = gs * np.sin(T) * np.cos(P)
            Y = gs * np.sin(T) * np.sin(P)
            Z = gs * np.cos(T)

            fg = plt.figure()
            ax = fg.gca(projection='3d')

            surf = ax.plot_surface(X, Y, Z, cmap=cm.jet,
                                   rstride=1, cstride=1,
                                   linewidth=0, antialiased=False, shade=True)

            fg.colorbar(surf, shrink=0.5, aspect=5)
            plt.show()


class Directivity2D(FarZone):
    pass

class Directivity3D(FarZone):
    pass

class FresnelPlane:
    def __init__(self, t, axis, r0, wxy, Nxy):
        if axis == 'X':
            rt = np.matrix([
                [1, 0, 0],
                [0, np.cos(t), -np.sin(t)],
                [0, np.sin(t), np.cos(t)]
            ])
        elif axis == 'Y':
            rt = np.matrix([
                [np.cos(t), 0, np.sin(t)],
                [0, 1, 0],
                [-np.sin(t), 0, np.cos(t)]
            ])
        elif axis == 'Z':
            rt = np.matrix([
                [np.cos(t), -np.sin(t), 0],
                [np.sin(t), np.cos(t), 0],
                [0, 0, 1]
            ])
        else:
            rt = np.matrix([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        wx, wy = wxy
        Nx, Ny = Nxy
        ox = np.linspace(-wx/2, wx/2, Nx)
        oy = np.linspace(-wy/2, wy/2, Ny)
        self.allpos = []
        self.alldat = np.empty(Nx*Ny, dtype=EfieldResult)
        self.lock = threading.Lock()
        self.size = Nx*Ny
        self.Nx = Nx
        self.Ny = Ny
        self.ox = ox
        self.oy = oy

        for y in oy:
            for x in ox:
                oxyz = np.matrix([x, y, r0])
                oxyz = np.transpose(oxyz)
                xyz = rt * oxyz
                R, T, P = car2sph(xyz.item(0), xyz.item(1), xyz.item(2))
                self.allpos.append((R, T, P))

    def __len__(self):
        return self.size

    def set_results(self, tsk):
        with self.lock:
            b, e = tsk.get_old_idx()
            for i in range(b, e):
                self.alldat.put(i, tsk.get_results()[i-b])

    def assign_task(self, mp=200):
        if len(self) <= mp:
            #pos = [(self.R, t, p) for t in self.col for p in self.row]
            return [Task((0, len(self)), self.allpos)]
        else:
            tsk = []
            b, e = 0, 0
            pos = []

            for i in range(len(self)):
                pos.append(self.allpos[i])
                e += 1
                if e - b == mp:
                    tsk.append(Task((b, e), pos))
                    pos = []
                    b = e
            tsk.append(Task((b, e), pos))
            return tsk

    def post_process(self, integ, fig=False):
        #fields = np.reshape(self.alldat, (self.Ny, self.Nx))
        mag, phase = [], []
        for dat in self.alldat:
            Ex, Ey, Ez = dat.get_car_field()
            phase.append(np.angle(Ey))
            mag.append(np.abs(np.sqrt(Ez**2 + Ey**2 + Ez**2)))

        mag = np.reshape(mag, (self.Ny, self.Nx))
        phase = np.reshape(phase, (self.Ny, self.Nx))

        if fig:
            plt.figure()
            plt.pcolor(self.ox, self.oy, mag)
            plt.show()

            plt.figure()
            plt.pcolor(self.ox, self.oy, phase)
            plt.show()
        return mag, phase


if __name__ == '__main__':
    #g2d = Gain2D(np.deg2rad(0), 100.)
    f2d = FresnelPlane(0.0, '0', 1.0, (0.5, 0.5), (50, 50))
    tsks = f2d.assign_task()

    for (i, tsk) in list(enumerate(tsks)):
        print('in task {}'.format(i))
        for (r, t, p) in tsk:
            print('r={},t={},p={}'.format(r, t, p))
        print('\n')


