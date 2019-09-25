# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve
import numpy as np

TYPE = {1: 'Motorcycle', 2: 'Auto', 3: 'Truck'}
TYPE2COLOR = {1: 'r', 2: 'g', 3: 'b'}


class Trajectroy(object):
    def __init__(self):
        self.tseries = []
        self.lxseries = []
        self.lyseries = []
        self.gxseries = []
        self.gyseries = []
        self.vseries = []


class Vehicle(object):
    def __init__(self, id, type, length, width):
        self.id = id
        self.type = type
        self.length = length
        self.width = width
        self.lx = 0
        self.ly = 0
        self.gx = 0
        self.gy = 0
        self.traj = Trajectroy()


class Frame(object):
    def __init__(self, id):
        self.id = id
        self.vehs = []


def main():
    # vehs = load_data_by_vehicle('./data/trajectories-0750am-0805am-test.csv')
    # gen_figure(vehs)
    frames, minframe, maxframe = load_data_by_frame('./data/trajectories-0750am-0805am.csv')
    gen_animation(frames, './data/US-101.png', './data/US-101.pgw')


def load_data_by_vehicle(dir):
    print('Reading...')
    data = pd.read_csv(dir)

    print('Generating vehicles...')
    vehs = []
    curVehID = None
    for index, row in data.iterrows():
        if index % 10000 == 0:
            print('\r%d %%' % int(index / data.shape[0] * 100), end='')
        elif index + 1 == data.shape[0]:
            print('\r100 %')

        if row['Vehicle_ID'] != curVehID:
            veh = Vehicle(id=row['Vehicle_ID'],
                          type=row['v_Class'],
                          length=row['v_Length'],
                          width=row['v_Width'])
            vehs.append(veh)
        vehs[-1].traj.tseries.append(row['Frame_ID'] * 0.1)  # 100 ms per frame
        vehs[-1].traj.lxseries.append(row['Local_X'])
        vehs[-1].traj.lyseries.append(row['Local_Y'])
        vehs[-1].traj.gxseries.append(row['Global_X'])
        vehs[-1].traj.gyseries.append(row['Global_Y'])
        vehs[-1].traj.vseries.append(row['v_Vel'])
        curVehID = row['Vehicle_ID']

    return vehs


def load_data_by_frame(dir):
    print('Reading...')
    data = pd.read_csv(dir)
    data.sort_values('Frame_ID', inplace=True)
    data.reset_index(inplace=True)
    minframe = data['Frame_ID'].min()
    maxframe = data['Frame_ID'].max()

    print('Generating frames...')
    frames = []
    curFrameID = None
    for index, row in data.iterrows():
        if index % 10000 == 0:
            print('\r%d %%' % int(index / data.shape[0] * 100), end='')
        elif index + 1 == data.shape[0]:
            print('\r100 %')

        if row['Frame_ID'] != curFrameID:
            frame = Frame(id=row['Frame_ID'])
            frames.append(frame)
        veh = Vehicle(id=row['Vehicle_ID'],
                      type=row['v_Class'],
                      length=row['v_Length'],
                      width=row['v_Width'])
        veh.lx = row['Local_X']
        veh.ly = row['Local_Y']
        veh.gx = row['Global_X']
        veh.gy = row['Global_Y']
        frames[-1].vehs.append(veh)
        curFrameID = row['Frame_ID']

    return (frames, minframe, maxframe)


def gen_figure(vehs):
    print('Generating figure...')
    fig = plt.figure(figsize=[25.6, 4.8])
    ax = fig.add_subplot(111)

    count = 0
    for veh in vehs:
        if count % 100 == 0:
            print('\r%d %%' % int(count / len(vehs) * 100), end='')
        elif count + 1 == len(vehs):
            print('\r100 %')

        ax.plot(veh.traj.tseries, veh.traj.lyseries,
                color=TYPE2COLOR[veh.type],
                label=TYPE[veh.type],
                linewidth=1)
        ax.set_title('Trajectory lines')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (feet)')
        ax.grid(True, linestyle='--')

        handle, label = ax.get_legend_handles_labels()
        handleout = []
        labelout = []
        for h, l in zip(handle, label):
            if l not in labelout:
                labelout.append(l)
                handleout.append(h)
        ax.legend(handleout, labelout)

        count += 1
    fig.show()


def gen_animation(frames, pngdir, pgwdir):
    print('Generating animation...')
    # paras for coordinate transform
    with open(pgwdir, 'r', encoding='utf-8') as f:
        A = float(f.readline().strip('\n'))
        D = float(f.readline().strip('\n'))
        B = float(f.readline().strip('\n'))
        E = float(f.readline().strip('\n'))
        C = float(f.readline().strip('\n'))
        F = float(f.readline().strip('\n'))

    # init basemap
    fig = plt.figure()
    ax = fig.add_subplot(111)
    basemap = plt.imread(pngdir)
    scatters = []

    def _init():
        ax.imshow(basemap)
        ax.axis('off')

    def _update(frame):
        # print(frame.id)
        if frame.id % 10 == 0:
            print('\r%d %%' % int(frame.id / len(frames) * 100), end='')
        elif frame.id + 1 == len(frames):
            print('\r100 %')

        ax.set_title('Frame: %d' % int(frame.id))

        for scatter in scatters:
            scatter.remove()
            del scatter
        scatters[:] = []

        for veh in frame.vehs:
            xy = _world2pixel(A, B, C, D, E, F, (veh.gx - veh.length, veh.gy - 0.5 * veh.width))
            # width = veh.length
            # height = veh.width
            # ax.add_patch(pat.Rectangle(xy, width, height, color=TYPE2COLOR[veh.type]))
            scatter = ax.scatter(xy[0], xy[1], color=TYPE2COLOR[veh.type], label=TYPE[veh.type])
            scatters.append(scatter)

    handles = []
    handles.append(pat.Patch(color=TYPE2COLOR[1], label=TYPE[1]))
    handles.append(pat.Patch(color=TYPE2COLOR[2], label=TYPE[2]))
    handles.append(pat.Patch(color=TYPE2COLOR[3], label=TYPE[3]))
    ax.legend(handles=handles)

    ani = FuncAnimation(fig, _update, frames=frames[:2000], init_func=_init)
    ani.save('Trajectory_animation.gif', writer='pillow', fps=30)


def _world2pixel(A, B, C, D, E, F, worldcoor):
    aa = np.array([[A, B], [D, E]])
    bb = np.array([worldcoor[0] - C, worldcoor[1] - F])
    pixelcoor = solve(aa, bb)
    return (pixelcoor[0], pixelcoor[1])


if __name__ == '__main__':
    main()
