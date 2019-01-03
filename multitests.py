from hornpattern import get_horn_input_power, get_default_pyramidal_horn, PyramidalHorn
import numpy as np
from sources import Source
from arrayinfo import RAInfo
from rautils import ideal_ref_unit, waveimpd, dB, sph2car
from rasolver import RASolver
from ratasks import Gain2D, Gain3D
import matplotlib.pyplot as plt


def test_offset_feed():
    freq = 6.0e9
    cell_sz = 25 / 1000.
    scale = 20
    horn = get_default_pyramidal_horn(freq, E0=1.0)
    integ = 0.07083
    showFig = True
    abg = (np.deg2rad(180), np.deg2rad(180), np.deg2rad(0))
    src = Source()
    fdr = 1.0
    fpos_offset = (0, cell_sz*2, cell_sz*scale*fdr)
    fpos = (0, 0, cell_sz*scale*fdr)
    src.append(horn, abg, fpos_offset, dir='parallel')

    tp = [(np.deg2rad(30), 0)]
    bs = 2
    arr = RAInfo(src, cell_sz, (scale, scale), ('pencil', tp), lambda p:ideal_ref_unit(p, bits=bs))
    solver = RASolver(arr)
    phi = np.deg2rad(0)
    tsk1 = Gain2D(phi, 300)
    #tsk1 = Gain3D(150, 150)
    solver.append_task(tsk1)
    solver.run()
    tsk1.post_process(integ, showFig)



def line_feed_array():
    freq = 10.0e9
    cell_sz = 15 / 1000.
    scale = 10
    abg = (np.deg2rad(180), np.deg2rad(180), np.deg2rad(0))
    integ = 1.2068
    showFig = True

    hz = cell_sz*scale*0.6

    fd_list = [(0.0, -cell_sz*2, hz),
               (0.0, -cell_sz, hz),
               (0.0, 0.0, hz),
               (0.0, cell_sz, hz),
               (0.0, cell_sz*2, hz)]

    src = Source()
    for i in range(len(fd_list)):
        src.append(get_default_pyramidal_horn(freq, E0=1.0, initp=0.0), abg, fd_list[i], dir='parallel')

    tp = [(0, 0)]
    bs = 1
    arr = RAInfo(src, cell_sz, (scale, scale), ('pencil', tp), lambda p:ideal_ref_unit(p, bits=bs))
    solver = RASolver(arr)
    phi = np.deg2rad(90)
    tsk1 = Gain2D(phi, 300)
    #tsk1 = Gain3D(200, 200)
    solver.append_task(tsk1)
    solver.run()
    tsk1.post_process(integ*5, showFig)

def linear_feed_pattern():
    freq = 10e9
    plist = [
        0, 45, 135, 202.5, 270, 270, 236.25, 247.5, 270, 247.5, 191.25, 112.5, 33.75, 348.75
    ]
    #plist = np.zeros(14)
    hlist = [
        PyramidalHorn(3.56, 5.08, 0.762, 0.3386, 1.524, 1.1854, 10.0, freq, initp) for initp in np.deg2rad(plist)
    ]

    ts = np.linspace(-np.pi/2, np.pi/2, 180)
    ps = np.deg2rad(90)
    R = 1.0

    cell_sz = 15.0 / 1000.
    pos = np.linspace(-cell_sz*6.5, cell_sz*6.5, 14)

    ds = []
    for t in ts:
        Et, Ep = 0., 0.
        for (i, h) in list(enumerate(hlist)):
            x, y, z = sph2car(R, t, ps)
            _, et, ep = h.efield_at_xyz(x, y-pos[i], z)
            print(et, ep)
            Et += et
            Ep += Ep
        Etotal = np.sqrt(Et*Et + Ep*Ep)
        d = np.abs(Etotal)*np.abs(Etotal)*R*R/(2*waveimpd)
        #print(d)
        ds.append(d)

    ds = dB(ds)
    plt.figure()
    plt.plot(np.rad2deg(ts), ds)
    plt.show()


if __name__ == '__main__':
    #test_offset_feed()
    #line_feed_array()
    linear_feed_pattern()
