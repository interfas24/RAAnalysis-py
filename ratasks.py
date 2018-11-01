import numpy as np
from rasolver import EfieldResult
from rautils import farR

class TaskState:
    Pending = 1
    Calculating = 2
    Finished = 3

class Task:
    def __init__(self, oid, pos):
        self.oid = oid
        self.pos = pos
        self.results = np.array(len(pos), dtype=EfieldResult)
        self.cnt = 0

    def __len__(self):
        return len(self.pos)

    def __iter__(self):
        self.cnt = 0
        return self

    def __next__(self):
        if self.cnt == len(self):
            raise StopIteration
        ret = self.pos[self.cnt]
        self.cnt += 1
        return ret

    def set_current_result(self, efield):
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

    def __len__(self):
        return self.nrow * self.ncol

    def __setitem__(self, key, value):
        if key > len(self):
            print('index exceeds')
            raise ValueError
        self.alldat[key] = value

    def __getitem__(self, item):
        return self.alldat[item]

    def set_results(self, tsk):
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


class Gain3D(FarZone):
    pass

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


