from rautils import create_array_pos, ideal_ref_unit, distance, make_v_mtx, R2F, F2R
from sources import Source
from hornpattern import get_default_pyramidal_horn
from beamforming import BeamForming
import numpy as np
import threading

import matplotlib.pyplot as plt

class RAInfo:

    def get_pxy(self):
        return self.px, self.py

    def get_Nxy(self):
        return self.Nx, self.Ny

    def get_dxy(self):
        return self.dx, self.dy

    def get_k0(self):
        return self.k0

    def __init__(self, src, cell_sz, scalexy, kv, func, shape='rect'):
        scalex, scaley = scalexy
        self.efield = []
        self.idx = 0
        self.px, self.py = cell_sz, cell_sz
        self.Nx, self.Ny = scalex, scaley
        self.dx, self.dy = cell_sz, cell_sz
        self.k0 = src.k0()

        xl, yl = create_array_pos(cell_sz, scalex, scaley, False)
        if shape == 'circle':
            if scalex != scaley:
                raise ValueError
            cpos = []
            for y in yl:
                for x in xl:
                    if np.sqrt(x**2 + y**2) <= scalex:
                        cpos.append((x, y))
        else:
            cpos = [(x, y) for y in yl for x in xl]

        key, val = kv
        fpos = [p for _, _, p, _ in src] if src.is_horn() else None
        bf = BeamForming(src.frequency(), xl, yl, fpos)
        if key == 'file':
            ret = []
            distro = open(val)
            for line in distro.readlines():
                sline = line.split()
                #v = [int(x) for x in sline]
                v = []
                for num in sline:
                    if int(num) == 1:
                        v.append(np.pi)
                    else:
                        v.append(0.0)
                ret.append(v)
            distro.close()
            phase = np.array(ret)
        elif key == 'selfdefine':
            phase = val
        elif key == 'pencil':
            phase = bf.form_pencil_beam(val, fpos != None)
        elif key == 'oam':
            tpm, beta = val
            phase = bf.form_oam_beam(tpm, beta, fpos != None)
        elif key == 'foci':
            phase = bf.form_focal_beam(val, fpos != None)
        else:
            print('Error phase distribution')
            raise ValueError

        allp = phase.flatten()
        #TODO: circle board

        sp = func(allp)

        # iter on sources
        for (i, (x, y)) in list(enumerate(cpos)):
            erx, ery = 0j, 0j
            for s, abg, pos, dir in src:
                dis = distance(pos, (0., 0., 0.))
                fpt = R2F(*abg) * make_v_mtx(x, y, 0.0)

                if dir == 'origin':
                    fpt[2][0] += dis
                elif dir == 'parallel':
                    fpt[0][0] -= pos[0]
                    fpt[1][0] -= pos[1]
                    fpt[2][0] += pos[2]

                Exyz, _, _ = s.efield_at_xyz(fpt.item(0), fpt.item(1), fpt.item(2))
                in_efield = F2R(*abg) * Exyz
                erx += in_efield.item(0)
                ery += in_efield.item(1)
            tc = np.matrix([
                [1, 0],
                [0, 1]
            ])
            s11, s12, s21, s22 = sp[i]
            d12 = tc*np.matrix([[erx], [ery]])
            a1 = s11*d12.item(0) + s12*d12.item(1)
            a2 = s21*d12.item(0) + s22*d12.item(1)
            axy = tc*np.matrix([[a1], [a2]])
            self.efield.append((axy.item(0), axy.item(1)))

        self.lock = threading.Lock()

    def __len__(self):
        return len(self.efield)

    def __iter__(self):
        with self.lock:
            self.idx = 0
            return self

    def __next__(self):
        with self.lock:
            if self.idx == len(self):
                raise StopIteration
            ret = self.efield[self.idx]
            self.idx += 1
            return ret


if __name__ == '__main__':
    freq = 5e9
    cell_sz = 30. / 1000.
    scale = 20

    abg = (np.deg2rad(180), np.deg2rad(180), np.deg2rad(0))
    src = Source()
    src.append(get_default_pyramidal_horn(freq), abg, (0., 0., 0.5))
    tp = [(np.deg2rad(0), np.deg2rad(0))]
    tpm = [(np.deg2rad(0), np.deg2rad(0), 1)]
    foci = [(0, 0, 0.8, 1.0)]

    arr = RAInfo(src, cell_sz, (scale, scale), ('oam', (tpm, np.deg2rad(10))), ideal_ref_unit)
    for (ex, ey) in arr:
        print(ex, ey)