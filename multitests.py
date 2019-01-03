from hornpattern import get_horn_input_power, get_default_pyramidal_horn, PyramidalHorn
import numpy as np
from sources import Source
from arrayinfo import RAInfo
from rautils import ideal_ref_unit, waveimpd, dB, sph2car, R2F, make_v_mtx
from rasolver import RASolver
from ratasks import Gain2D, Gain3D, FresnelPlane
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
    #plist = [
        #0, 45, 135, 202.5, 270, 270, 236.25, 247.5, 270, 247.5, 191.25, 112.5, 33.75, 348.75
    #]
    plist = np.zeros(1)
    hlist = [
        PyramidalHorn(3.56, 5.08, 0.762, 0.3386, 1.524, 1.1854, 10.0, freq, initp) for initp in np.deg2rad(plist)
    ]

    abg = (np.deg2rad(180), np.deg2rad(180), np.deg2rad(0))
    ts = np.linspace(-np.pi/2, np.pi/2, 300)
    ps = np.deg2rad(0)
    R = 1.0

    cell_sz = 180 / 1000.
    #pos = np.linspace(-cell_sz*6.5, cell_sz*6.5, 14)
    #pos = [-cell_sz/2, cell_sz/2]
    pos = [0.0]

    ds = []
    for t in ts:
        #Et, Ep = 0j, 0j
        """
        #for (i, h) in list(enumerate(hlist)):
            x, y, z = sph2car(R, t, ps)
            fpt = R2F(*abg) * make_v_mtx(x, y, 0.0)
            fpos = (0.0, pos[i], z)
            fpt[0][0] -= fpos[0]
            fpt[1][0] -= fpos[1]
            fpt[2][0] += fpos[2]
            _, et, ep = h.efield_at_xyz(fpt.item(0), fpt.item(1), fpt.item(2))
            #_, et, ep = h.efield_at_rtp(R, t, ps)
            #Et += et
            #Ep += Ep
        """
        x, y, z = sph2car(R, t, ps)
        #_, Et1, Ep1 = hlist[0].efield_at_xyz(x-pos[0], y, z)
        #_, Et2, Ep2 = hlist[1].efield_at_xyz(x-pos[1], y, z)
        #Et = Et1 + Et2
        #Ep = Ep1 + Ep2
        _, Et, Ep = hlist[0].efield_at_xyz(x, y, z)
        Etotal = np.sqrt(Et*Et + Ep*Ep)
        d = np.abs(Etotal)*np.abs(Etotal)*R*R/(2*waveimpd)
        ds.append(d)

    ds = dB(ds)
    plt.figure()
    plt.plot(np.rad2deg(ts), ds)
    plt.show()

def calc_focal_2bit():
    freq = 5.8e9
    cell_sz = 30 / 1000.
    scale = 10
    fdr = 0.9
    hz = cell_sz*scale*fdr
    abg = (np.deg2rad(180), np.deg2rad(180), np.deg2rad(0))
    showFig = True

    integ = 40.59829
    horn = get_default_pyramidal_horn(freq)
    #print(get_horn_input_power(horn))
    pos = (0.0, 0.0, hz)
    src = Source()
    src.append(horn, abg, pos)
    foki = [(-0.3, 0.3, 1.5, 1.0), (0.3, -0.3, 1.5, 1.0)]
    bs = 2
    arr = RAInfo(src, cell_sz, (scale, scale), ('foci', foki), lambda p:ideal_ref_unit(p, bits=bs))

    solver = RASolver(arr)
    tsk1 = FresnelPlane(0, 'xx', 1.5, (1.2, 1.2), (200, 200))
    p = np.deg2rad(125)
    tsk2 = Gain2D(p, 300)
    solver.append_task(tsk1)
    #solver.append_task(tsk2)
    solver.run()
    tsk1.post_process(integ, showFig)
    #tsk2.post_process(integ, showFig)


if __name__ == '__main__':
    #test_offset_feed()
    #line_feed_array()
    linear_feed_pattern()
    #calc_focal_2bit()
