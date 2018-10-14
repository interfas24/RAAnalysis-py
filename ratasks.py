import numpy as np

class TaskState:
    Pending = 1
    Calculating = 2
    Finished = 3


class FarZone:
    def __init__(self, row, col):
        self.state = TaskState.Pending
        self.nrow = len(row)
        self.ncol = len(col)
        self.alldat = np.ndarray(shape=(1, self.nrow*self.ncol))

    def __len__(self):
        return self.nrow * self.ncol

    def __setitem__(self, key, value):
        if key > len(self):
            print('index exceeds')
            raise ValueError
        self.alldat[0][key] = value

    def __getitem__(self, item):
        return self.alldat[0][item]


class Gain2D(FarZone):
    def __init__(self, phi, ntheta):
        ts = np.linspace(-np.pi/2, np.pi/2, ntheta)
        FarZone.__init__(self, [phi], ts)

class Gain3D(FarZone):
    pass

class Directivity2D(FarZone):
    pass

class Directivity3D(FarZone):
    pass

class FresnelPlane:
    pass