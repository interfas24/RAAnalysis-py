from hornpattern import get_horn_input_power, get_default_pyramidal_horn, PyramidalHorn
import numpy as np
from sources import Source
from arrayinfo import RAInfo
from rautils import ideal_ref_unit, waveimpd, dB, sph2car, R2F, make_v_mtx
from rasolver import RASolver
from ratasks import Gain2D, Gain3D, FresnelPlane, OnAxisLine, PhiCutPlane
import matplotlib.pyplot as plt
import csv


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

def focal_2bit_calculation():
    freq = 5.8e9
    cell_sz = 30 / 1000.
    scale = 10
    fdr = 0.9
    hz = cell_sz*scale*fdr
    abg = (np.deg2rad(180), np.deg2rad(180), np.deg2rad(0))
    showFig = True

    integ = 1.039    # 1.6, 5.8e9
    horn = get_default_pyramidal_horn(freq, E0=1.6)
    pos = (0.0, 0.0, hz)
    src = Source()
    src.append(horn, abg, pos)

    focuses = (
        [(1.0, 1.0, 1.0, 1.0)],
        [(-0.3, 0.3, 1.5, 1.0), (0.3, -0.3, 1.5, 1.0)],
        [(0.3, 0.3, 2.0, 1.0)]
    )
    hs = (
        1.0,
        1.5,
        2.0
    )
    planes = (
        (3.0, 3.0),
        (1.2, 1.2),
        (1.2, 1.2)
    )
    points = (
        (301, 301),
        (121, 121),
        (121, 121)
    )
    ps = (
        np.deg2rad(45),
        np.deg2rad(135),
        np.deg2rad(45)
    )

    Nt, Np = 181, 181

    for idx in range(3):
        bs = 2
        arr = RAInfo(src, cell_sz, (scale, scale), ('foci', focuses[idx]), lambda p:ideal_ref_unit(p, bits=bs))

        solver = RASolver(arr)
        tsk1 = FresnelPlane(0, 'xx', hs[idx], planes[idx], points[idx])
        tsk2 = Gain2D(ps[idx], Nt, freq)
        tsk3 = Gain3D(Np, Nt)

        solver.append_task(tsk1)
        solver.append_task(tsk2)
        solver.append_task(tsk3)

        solver.run()

        tsk1.post_process(integ, showFig, exfn='experiment/2bit/case{}_plane_theo.fld'.format(idx+1))
        tsk2.post_process(integ, showFig, exfn='experiment/2bit/case{}_2d_theo.csv'.format(idx+1))
        tsk3.post_process(integ, showFig, exfn='experiment/2bit/case{}_3d_theo.csv'.format(idx+1))


def my_unit(pha, amp=1.0, bits=None, pol='Y'):
    ret = []
    for i in range(len(pha)):
        sp = pha[i]
        if bits != None:
            step = np.pi*2/(2**bits)
            sp = int(pha[i]/step) * step

        if pol == 'X':
            sparam = (amp*np.exp(1j*sp), 0j, 0j, 1+0j)
        elif pol == 'Y':
            sparam = (1+0j, 0j, 0j, amp*np.exp(1j*sp))
        elif pol == 'XY':
            sparam = (amp*np.exp(1j*sp), 0j, 0j, amp*np.exp(1j*sp))
        else:
            sparam = (amp*np.exp(1j*sp), 0j, 0j, amp*np.exp(1j*(sp-np.pi/2.0)))

        ret.append(sparam)

    return ret


def RA_12x12():
    freq = 5.0e9
    cell_sz = 33 / 1000.
    scale = 12
    fdr = 0.9
    hz = cell_sz*scale*fdr
    abg = (np.deg2rad(180), np.deg2rad(180), np.deg2rad(0))
    showFig = True

    integ = 30.17114    # 10.0, 5.0e9
    horn = get_default_pyramidal_horn(freq)
    pos = (0.0, 0.0, hz)
    src = Source()
    src.append(horn, abg, pos)

    bs = 2
    tmp = [(np.deg2rad(0), np.deg2rad(0))]
    arr = RAInfo(src, cell_sz, (scale, scale), ('pencil', tmp), lambda p:ideal_ref_unit(p, bits=bs, amp=0.8))

    solver = RASolver(arr)
    tsk1 = Gain2D(np.deg2rad(0), 181)

    solver.append_task(tsk1)

    solver.run()

    tsk1.post_process(integ, showFig, exfn='experiment/12x12/xoz-theo.csv')


def circle_unit_data_table():
    fn = 'experiment/s11tetm-circle.csv'
    data_table = []
    csvf = open(fn)
    sparam = csv.reader(csvf)
    for row in sparam:
        row = [float(s) for s in row]
        d = []
        for i in range(4):
            d.append(row[i*2] + 1j*row[i*2+1])
        data_table.append(d)
        #print(d)
    rad_list = []
    for s in data_table:
        rad_list.append(np.angle(s[0]))
    #print(rad_list)
    csvf.close()
    return data_table, rad_list

sparam_dt, s11_rads = circle_unit_data_table()

def get_sparam_line_idx(rad):
    rad -= np.pi
    if rad > s11_rads[0]:
        return 0
    elif rad < s11_rads[-1]:
        return len(s11_rads) - 1
    for i in range(len(s11_rads)-1):
        if rad < s11_rads[i] and rad > s11_rads[i+1]:
            mid = (s11_rads[i] + s11_rads[i+1]) / 2.0
            if rad >= (mid):
                return i
            else:
                return i+1

def circle_unit(pha):
    ret = []
    for i in range(len(pha)):
        p = pha[i]
        idx = get_sparam_line_idx(p)
        ret.append(sparam_dt[idx])

    return ret



def test_fresnel_plane():
    freq = 6.0e9
    cell_sz = 25 / 1000.
    lmbd = 3e8 / freq
    scale = 20
    fdr = 0.8
    hz = cell_sz*scale*fdr
    horn = get_default_pyramidal_horn(freq)
    pos = (0.0, 0.0, hz)
    abg = (np.deg2rad(180), np.deg2rad(180), np.deg2rad(0))
    showFig = True

    far_z = 2 * (scale*cell_sz*np.sqrt(2)) ** 2 / lmbd
    focal = [(0.0, 0.0, 5.0, 1.0)]

    src = Source()
    src.append(horn, abg, pos)

    arr = RAInfo(src, cell_sz, (scale, scale), ('foci', focal), circle_unit)

    solver = RASolver(arr, type='fresnel')

    tsk = FresnelPlane(0, 'xx', 5.0, (1.0, 1.0), (41, 41))
    solver.append_task(tsk)

    solver.run()

    tsk.post_process(showFig, None)


def test_fresnel_onaxis():
    freq = 6.0e9
    cell_sz = 25 / 1000.
    lmbd = 3e8 / freq
    scale = 20
    fdr = 0.8
    hz = cell_sz*scale*fdr
    horn = get_default_pyramidal_horn(freq)
    pos = (0.0, 0.0, hz)
    abg = (np.deg2rad(180), np.deg2rad(180), np.deg2rad(0))
    showFig = True
    far_z = 2 * (scale*cell_sz*np.sqrt(2)) ** 2 / lmbd

    focal_rio = 0.1
    focal = [(0.0, 0.0, focal_rio*far_z, 1.0)]

    src = Source()
    src.append(horn, abg, pos)

    arr = RAInfo(src, cell_sz, (scale, scale), ('foci', focal), circle_unit)

    solver = RASolver(arr, type='fresnel')

    plane_rio = np.linspace(0.01, 0.5, 200)
    zlist = [pr*far_z for pr in plane_rio]

    tsk = OnAxisLine(zlist, plane_rio)
    solver.append_task(tsk)

    solver.run()

    tsk.post_process(showFig, None)


def test_fresnel_phicut():
    freq = 6.0e9
    cell_sz = 25 / 1000.
    lmbd = 3e8 / freq
    scale = 20
    fdr = 0.8
    hz = cell_sz*scale*fdr
    horn = get_default_pyramidal_horn(freq)
    pos = (0.0, 0.0, hz)
    abg = (np.deg2rad(180), np.deg2rad(180), np.deg2rad(0))
    showFig = True

    far_z = 2 * (scale*cell_sz*np.sqrt(2)) ** 2 / lmbd
    focal = [(0.5, 0.5, 1.0, 1.0), (-0.5, -0.5, 1.0, 1.0)]

    src = Source()
    src.append(horn, abg, pos)

    arr = RAInfo(src, cell_sz, (scale, scale), ('foci', focal), circle_unit)

    solver = RASolver(arr, type='fresnel')

    tsk = PhiCutPlane((2.0, 1.5), (21, 21), phi=np.deg2rad(45))

    solver.append_task(tsk)

    solver.run()

    tsk.post_process(showFig, None)


if __name__ == '__main__':
    test_fresnel_phicut()
    #test_fresnel_plane()
    #test_fresnel_onaxis()
    #test_offset_feed()
    #line_feed_array()
    #linear_feed_pattern()
    #focal_2bit_calculation()
    #RA_12x12()

