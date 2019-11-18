import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import argparse


Experience = namedtuple(
    # TODO - consider multi-frame state to include time related information
    'Experience',
    ('state', 'bin_idx', 'action', 'next_state', 'reward')
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.bin_idx)
    t2 = t2.view(-1, 2)
    t3 = torch.cat(batch.action)
    t4 = torch.cat(batch.reward)
    t5 = torch.cat(batch.next_state)

    return (t1, t2, t3, t4, t5)


def plot_sweep(data):
    plt.scatter(data[:, 1], data[:, 2])
    plt.show()
    return


def get_inf_corner(data):
    return data[-1, 1], data[-1, 2]


def get_sup_corner(data):
    return data[0, 1], data[0, 2]


def center_data(data):
    x_inf, y_inf = get_inf_corner(data)
    data[:, 1] -= x_inf
    data[:, 2] -= y_inf
    return data

def force_into_grid(data, cols=11, col_range=200):
    data = data[:col_range*cols, :]
    data_y_shift = np.nanmin(data[:, 2])
    col_dist = np.abs(np.nanmean(data[:col_range, 2]) - np.nanmean(data[(cols-1)*col_range:cols*col_range, 2]) / cols)
    print(col_dist)
    for i in range(cols):
        data[i*col_range:(i+1) * col_range, 1] = i*col_dist
        data[i*col_range:(i+1)*col_range, 2] -= np.nanmin(data[i*col_range:(i+1)*col_range, 2])
    data[:, 2] += data_y_shift
    plot_sweep(data)
    return data


def plot_sweep(data):
    plt.scatter(data[:, 1], data[:, 2])
    plt.show()


def def_step_size(steps_x, steps_y, margins, data):
    mins = np.nanmin(data[:,1:3], axis=0)
    maxs = np.nanmax(data[:,1:3], axis=0)
    return (maxs - mins - 2 * margins)/[steps_x-1, steps_y-1]


def load_data(path):
    frame_coords = np.genfromtxt(path, dtype=float, delimiter=',', names=True)
    header = frame_coords.dtype.names
    data = frame_coords.view((float, len(frame_coords.dtype.names)))
    if "tilting" in path:
        data = data[:, :7]
    else:
        data = data[:, :3]
    return header, data


def check_nop(actions, rewards, true_positive = True):
    actions = actions.detach().cpu().numpy()
    rewards = rewards.detach().cpu().numpy()
    nop_actions = (actions == 4)
    nop_rewards = rewards * nop_actions
    if(true_positive):
        return nop_rewards > 0
    return nop_rewards < 0 # + (rewards == -0.25)


def check_rewards(rewards, positive=True):
    rewards = rewards.detach().cpu().numpy()
    if(positive):
        return rewards >= 0
    return rewards < 0
