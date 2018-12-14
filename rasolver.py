from hornpattern import PyramidalHorn, get_default_pyramidal_horn, get_horn_input_power
from arrayinfo import RAInfo
from sources import PlaneWave, Source
from rautils import gsinc, ideal_ref_unit, dB, sph2car_mtx, make_v_mtx, farR
from ratasks import Gain2D, Gain3D, EfieldResult
import numpy as np
import matplotlib.pyplot as plt
import time
import concurrent.futures
import threading

"""
1. array info (cell size scale | horns)
2. get phase
3. get tetm
4. add task
5. calc
"""

class RASolver:

    def __init__(self, rainfo):
        self.rainfo = rainfo
        self.tasks = []
        self.lock = threading.Lock()

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

        return EfieldResult(E_theta, E_phi, (r, t, p))


    # add far zone task
    def append_task(self, tsk):
        self.tasks.append(tsk)

    # iterate on tasks, each task use multi-threading
    def run(self):
        for task in self.tasks:
            field_task = task.assign_task()
            start_time = time.clock()
            for tsk in field_task:
                for (r, t, p) in tsk:
                    tsk.set_current_result(self.__calc_one_point(r, t, p))
                task.set_results(tsk)
            print(time.clock()-start_time, "sec")

    # calc a task in a thread
    def __calc_one_task__(self, tsk):
        #with self.lock:
            for (r, t, p) in tsk:
                tsk.set_current_result(self.__calc_one_point(r, t, p))
            return tsk

    # Bugs here
    def run_concurrency(self):
        for task in self.tasks:
            field_task = task.assign_task()
            start_time = time.clock()
            tsk_list = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as e:
                for tsk in field_task:
                    print(type(tsk))
                    e.submit(self.__calc_one_task__, tsk)
                    tsk_list.append(tsk)
            print(time.clock()-start_time, "sec")

            for t in tsk_list:
                task.set_results(t)

    def test(self):
        N = 300
        ts = np.linspace(np.deg2rad(-89), np.deg2rad(89), N)
        ps = np.deg2rad(0)

        res = []
        for t in ts:
            res.append(self.__calc_one_point(farR, t, ps))
        et = []
        integ = 22732.769823328235
        for x in res:
            et.append(x.get_gain(integ))
        et = dB(et)
        plt.figure()
        plt.plot(np.rad2deg(ts), et)
        plt.show()


def test1():
    freq = 5e9
    cell_sz = 30. / 1000.
    scale = 20

    abg = (np.deg2rad(180), np.deg2rad(180), np.deg2rad(0))
    src = Source()
    horn = get_default_pyramidal_horn(freq)
    src.append(horn, abg, (0., 0., 0.5))
    tp = [(np.deg2rad(0), np.deg2rad(0))]
    tpm = [(np.deg2rad(20), np.deg2rad(0), 1)]
    foci = [(0, 0, 0.8, 1.0)]

    #arr = RAInfo(src, cell_sz, (scale, scale), ('oam', (tpm, np.deg2rad(0))), ideal_ref_unit)
    arr = RAInfo(src, cell_sz, (scale, scale), ('pencil', tp), ideal_ref_unit)
    solver = RASolver(arr)
    solver.test()

def test2():
    freq = 5e9
    cell_sz = 30. / 1000.
    scale = 20

    abg = (np.deg2rad(180), np.deg2rad(180), np.deg2rad(0))
    src = Source()
    horn = get_default_pyramidal_horn(freq)
    src.append(horn, abg, (0., 0., 0.5))
    tp = [(np.deg2rad(0), np.deg2rad(0))]
    tpm = [(np.deg2rad(20), np.deg2rad(0), 1)]
    foci = [(0, 0, 0.8, 1.0)]

    #arr = RAInfo(src, cell_sz, (scale, scale), ('oam', (tpm, np.deg2rad(0))), ideal_ref_unit)
    arr = RAInfo(src, cell_sz, (scale, scale), ('pencil', tp), lambda p:ideal_ref_unit(p, bits=3))
    solver = RASolver(arr)
    solver.test()

def test_multi():
    freq = 5e9
    cell_sz = 30. / 1000.
    scale = 20

    abg = (np.deg2rad(180), np.deg2rad(180), np.deg2rad(0))
    src = Source()
    horn = get_default_pyramidal_horn(freq)
    src.append(horn, abg, (0., 0., cell_sz*scale))
    tp = [(np.deg2rad(0), np.deg2rad(0))]

    arr = RAInfo(src, cell_sz, (scale, scale), ('pencil', (tp)), ideal_ref_unit)
    solver = RASolver(arr)
    tsk1 = Gain2D(np.deg2rad(0), 300)
    tsk2 = Gain2D(np.deg2rad(90), 300)
    tsk3 = Gain3D(100, 100)
    #solver.append_task(tsk1)
    #solver.append_task(tsk2)
    solver.append_task(tsk3)
    solver.run()
    #solver.run_concurrency()
    #tsk1.post_process()
    #tsk2.post_process()
    tsk3.post_process(22732.769823328235, True)


if __name__ == '__main__':
    #test1()
    #test2()
    test_multi()
