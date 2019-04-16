import numpy as np
from rautils import farR, sph2car_mtx, make_v_mtx, dB, car2sph, waveimpd
import threading
import matplotlib.pyplot as plt
from matplotlib import cm
import cmath

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
        return np.abs(self.Etotal)*np.abs(self.Etotal)*r*r/(2*waveimpd)

    def get_gain(self, ipower):
        return 4*np.pi*np.abs(self.Etotal)*np.abs(self.Etotal) * farR * farR / (ipower * 2 * waveimpd)

    def get_rE(self, type='total'):
        (r, t, p) = self.Pos
        if type == 'total':
            return self.Etotal * r
        elif type == 'theta':
            return self.Etheta * r
        elif type == 'phi':
            return self.Ephi * r

class NearFieldResult:
    def __init__(self, Ex, Ey, Ez, pos):
        self.Ex = Ex
        self.Ey = Ey
        self.Ez = Ez
        self.Pos = pos
        self.Etotal = np.sqrt(Ex*Ex + Ey*Ey + Ez*Ez)

    def get_etotal(self):
        return self.Etotal


    def get_car_field(self):
        return self.Ex, self.Ey, self.Ez


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
    def __init__(self, row, col, freq=10e9):
        self.row = row
        self.col = col
        self.nrow = len(row)
        self.ncol = len(col)
        self.alldat = np.empty(self.nrow*self.ncol, dtype=EfieldResult)
        self.lock = threading.Lock()
        self.R = farR
        self.freq = freq

    def __len__(self):
        return self.nrow * self.ncol

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
    def __init__(self, phi, ntheta, freq=10e9):
        ts = np.linspace(-np.pi/2, np.pi/2, ntheta)
        super().__init__([phi], ts, freq)

    def post_process(self, integ, fig=False, exfn=None):
        fields = np.reshape(self.alldat, (self.nrow, self.ncol))
        gs = [x.get_gain(integ) for x in fields[0]]
        gs = dB(gs)

        if fig:
            plt.figure()
            plt.plot(self.col, gs)
            plt.ylim(-30, 40)
            plt.show()

        if exfn != None:
            afile = open(exfn, 'w')
            afile.write('Theta[deg],dB(GainTotal)[] - Freq=\'{}GHz\' Phi=\'{}deg\'\n'
                        .format(self.freq/1e9, np.rad2deg(self.row[0])))
            ts = np.rad2deg(self.col)
            for i in range(len(ts)):
                afile.write('{},{}\n'.format(ts[i], gs[i]))
            afile.close()


class Gain3D(FarZone):
    def __init__(self, nphi, ntheta, freq=1e9):
        self.ps = np.linspace(0, np.pi*2, nphi)
        self.ts = np.linspace(-np.pi/2, np.pi/2, ntheta)
        super().__init__(self.ps, self.ts, freq)

    def post_process(self, integ, fig=False, exfn=None):
        fields = np.reshape(self.alldat, (self.nrow, self.ncol))
        gs = np.ndarray(shape=fields.shape)
        for (id, field) in list(enumerate(fields)):
            g = dB([x.get_gain(integ) for x in field])
            gs[id,:] = np.array(g)

        if exfn != None:
            afile = open(exfn, 'w')
            afile.write('Phi[deg],Theta[deg],dB(GainTotal)\n')
            ps = np.rad2deg(self.row)
            ts = np.rad2deg(self.col)
            for i in range(self.ncol):
                for j in range(self.nrow):  #j phi
                    afile.write('{},{},{}\n'.format(ps[j], ts[i], gs[j, i]))
            afile.close()

        if fig:
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
            #ax = fg.gca(projection='3d')
            ax = fg.gca()

            surf = ax.plot_surface(X, Y, Z, cmap=cm.jet,
                                   rstride=1, cstride=1,
                                   linewidth=0, antialiased=False, shade=True)

            fg.colorbar(surf, shrink=0.5, aspect=5)
            plt.show()


class rE3D(FarZone):
    def __init__(self, nphi, ntheta, freq=1e9):
        self.ps = np.linspace(0, np.pi*2, nphi)
        self.ts = np.linspace(-np.pi/2, np.pi/2, ntheta)
        super().__init__(self.ps, self.ts, freq)

    def post_process(self, type='total', fig=False, mfn=None, pfn=None):
        fields = np.reshape(self.alldat, (self.nrow, self.ncol))
        gs = np.ndarray(shape=fields.shape, dtype=complex)

        for (id, field) in list(enumerate(fields)):
            g = [x.get_rE(type) for x in field]
            gs[id,:] = np.array(g)


        if mfn != None:
            afile = open(mfn, 'w')
            afile.write('Phi[deg],Theta[deg],mag(rETotal)[V]\n')
            ps = np.rad2deg(self.row)
            ts = np.rad2deg(self.col)
            for i in range(self.ncol):
                for j in range(self.nrow):  #j phi
                    afile.write('{},{},{}\n'.format(ps[j], ts[i], np.abs(gs[j, i])))
            afile.close()

        if pfn != None:
            afile = open(pfn, 'w')
            afile.write('Phi[deg],Theta[deg],phase(rETotal)[deg]\n')
            ps = np.rad2deg(self.row)
            ts = np.rad2deg(self.col)
            for i in range(self.ncol):
                for j in range(self.nrow):  #j phi
                    afile.write('{},{},{}\n'.format(ps[j], ts[i], np.angle(gs[j, i], deg=True)))
            afile.close()

        if fig:
            ps = self.row
            ts = self.col
            T, P = np.meshgrid(ts, ps)

            plt.figure()
            plt.pcolor(np.sin(T)*np.cos(P), np.sin(T)*np.sin(P), np.abs(gs))
            plt.colorbar()

            plt.figure()
            plt.pcolor(np.sin(T)*np.cos(P), np.sin(T)*np.sin(P), np.angle(gs, deg=True))
            plt.colorbar()

            plt.show()


class Directivity2D(FarZone):
    def __init__(self, phi, ntheta, freq=10e9):
        ts = np.linspace(-np.pi/2, np.pi/2, ntheta)
        super().__init__([phi], ts, freq)

    def post_process(self, integ, fig=False, exfn=None):
        fields = np.reshape(self.alldat, (self.nrow, self.ncol))
        gs = [x.get_directivity() for x in fields[0]]
        gs = dB(gs)

        if fig:
            plt.figure()
            plt.plot(self.col, gs)
            plt.ylim(-30, 40)
            plt.show()

        if exfn != None:
            afile = open(exfn, 'w')
            afile.write('Theta[deg],dB(GainTotal)[] - Freq=\'{}GHz\' Phi=\'{}deg\'\n'
                        .format(self.freq/1e9, np.rad2deg(self.row[0])))
            ts = np.rad2deg(self.col)
            for i in range(len(ts)):
                afile.write('{},{}\n'.format(ts[i], gs[i]))
            afile.close()

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
        self.alldat = np.empty(Nx*Ny, dtype=NearFieldResult)
        self.lock = threading.Lock()
        self.size = Nx*Ny
        self.Nx = Nx
        self.Ny = Ny
        self.ox = ox
        self.oy = oy
        self.r0 = r0

        for y in oy:
            for x in ox:
                oxyz = np.matrix([x, y, r0])
                oxyz = np.transpose(oxyz)
                xyz = rt * oxyz
                self.allpos.append((xyz.item(0), xyz.item(1), xyz.item(2)))

    def __len__(self):
        return self.size

    def set_results(self, tsk):
        with self.lock:
            b, e = tsk.get_old_idx()
            for i in range(b, e):
                self.alldat.put(i, tsk.get_results()[i-b])

    def assign_task(self, mp=200):
        if len(self) <= mp:
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

    def post_process(self, fig=False, mfn=None):
        mag, phase = [], []
        for dat in self.alldat:
            total = dat.get_etotal()
            ex, ey, ez = dat.get_car_field()
            phase.append(np.angle(ey))
            mag.append(np.abs(total))

        mag = np.reshape(mag, (self.Nx, self.Ny))

        if fig:
            plt.figure()
            plt.pcolor(self.ox, self.oy, mag, cmap='jet')
            plt.colorbar()
            plt.show()


        if mfn != None:
            afile = open(mfn, 'w')
            afile.write('X[m],X[m],mag(ETotal)[V/m]\n')
            for i in range(len(mag)):
                for j in range(len(mag[0])):  #j phi
                    afile.write('{},{},{}\n'.format(self.ox[i], self.oy[j], mag[i,j]))
            afile.close()


class OnAxisLine:
    def __init__(self, zlist, zrio):
        self.allpos = [(0.0, 0.0, z) for z in zlist]
        self.alldat = np.empty(len(zlist), dtype=NearFieldResult)
        self.size = len(zlist)
        self.zlist = zlist
        self.zrio = zrio

    def __len__(self):
        return self.size

    def set_results(self, tsk):
        b, e = tsk.get_old_idx()
        for i in range(b, e):
            self.alldat.put(i, tsk.get_results()[i-b])

    def assign_task(self, mp=200):
        if len(self) <= mp:
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

    def post_process(self, fig=False, mfn=None):
        mag, phase = [], []
        for dat in self.alldat:
            total = dat.get_etotal()
            ex, ey, ez = dat.get_car_field()
            phase.append(np.angle(ey))
            mag.append(np.abs(total))

        if fig:
            plt.figure()
            plt.plot(self.zrio, mag)
            plt.show()

        if mfn != None:
            afile = open(mfn, 'w')
            afile.write('Z[m],mag(ETotal)[V/m]\n')
            for idx in range(len(self)):
                afile.write('{},{}\n'.format(self.zlist[idx], mag[idx]))
            afile.close()


if __name__ == '__main__':
    #g2d = Gain2D(np.deg2rad(0), 100.)
    f2d = FresnelPlane(0.0, '0', 1.0, (0.5, 0.5), (50, 50))
    tsks = f2d.assign_task()

    for (i, tsk) in list(enumerate(tsks)):
        print('in task {}'.format(i))
        for (r, t, p) in tsk:
            print('r={},t={},p={}'.format(r, t, p))
        print('\n')


