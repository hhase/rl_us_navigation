import torch
import random
import numpy as np
import scipy.signal
import scipy.ndimage
from PIL import Image
import torchvision.transforms as T
import utils.utility_functions as fun

np.set_printoptions(suppress=True)


class Patient():
    patient_counter = 0
    def __init__(self, name, bin_grid, frame_path, frames_per_vert_sweep, goal_coords, args, cluster=False):
        self.id = Patient.patient_counter
        Patient.patient_counter += 1
        self.name = name
        self.frame_path = frame_path
        self.frames_per_vert_sweep = frames_per_vert_sweep
        self.goal_coords = goal_coords
        self.grid = Grid(bin_grid, goal_coords, args, self.frame_path, cluster)


class Bin():
    def __init__(self, coords, contains_goal_state=False):
        self.coords = coords
        self.frame_idx = list()
        self.frames = None
        self.contains_goal_state = contains_goal_state
        self.rotations = None

    def distance(self, goal_coords):

        distance = np.sqrt(np.sum(np.power(self.coords - goal_coords, 2)))
        return distance


class Grid():
    def __init__(self, bin_grid, goal_state, args, frame_path, cluster=False):
        self.bin_array = bin_grid
        self.current_bin = self.get_random_bin()
        self.frame_path = frame_path
        self.cluster = cluster
        self.visits = np.zeros([args.steps_x, args.steps_y])
        self.goal_state = goal_state
        self.goal_coords = list()
        self.device = args.device
        self.load_frames_per_bin()
        self.reward_dict = {'goal_correct':     args.reward_correct_nop,
                            'goal_incorrect':   args.reward_false_nop,
                            'border_collision': args.reward_border,
                            'closer':           args.reward_move_closer,
                            'further':          args.reward_move_away}
        self.current_reward = 0
        self.action_space = {0: 'UP',
                             1: 'DOWN',
                             2: 'LEFT',
                             3: 'RIGHT',
                             4: 'NOP'}
        self.distance = self.current_bin.distance(self.goal_state)
        self.border_collisions = 0
        self.plotting = False

    def load_frames_per_bin(self):
        rows, cols = self.bin_array.shape
        frame_x_dim, frame_y_dim = self.get_processed_frame().shape[2:]
        for i in range(rows):
            for j in range(cols):
                self.bin_array[i, j].frames = torch.zeros([len(self.bin_array[0,0].frame_idx), frame_x_dim, frame_y_dim])
                for k in range(len(self.bin_array[i,j].frame_idx)):
                    self.bin_array[i,j].frames[k,:,:] = self.get_processed_frame(self.bin_array[i,j].frame_idx[k])

    def populate_grid(self, steps_x, steps_y, step_size, margins, data):
        grid = np.zeros((steps_x, steps_y), dtype=Bin)
        data_y_shift = np.nanmin(data[:, 2])
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                x_location = i * step_size[0] + margins[0]
                y_location = j * step_size[1] + margins[1]
                location = np.array([x_location, y_location + data_y_shift])
                coords = np.array([i, j])
                frame_idx = find_closest_frame(location, data)
                grid[i, j] = Bin(coords, frame_idx)
        return grid

    def get_random_bin(self):
        idx = self.bin_array.shape[0] * self.bin_array.shape[1]
        rand_idx = np.random.randint(idx)
        new_bin = self.bin_array[int(rand_idx%self.bin_array.shape[0]), int(rand_idx/self.bin_array.shape[0])]
        return new_bin

    def reset_grid(self):
        new_bin = self.get_random_bin()
        self.set_grid(new_bin.coords)

    def set_grid(self, coords):
        self.current_bin = self.bin_array[coords[0], coords[1]]
        self.distance = self.current_bin.distance(self.goal_state)

    def get_bin(self, coords):
        return self.bin_array[coords[0], coords[1]]

    def check_boundaries(self, new_coords):
        x_range, y_range = self.bin_array.shape
        if new_coords[0] in range(x_range) and new_coords[1] in range(y_range):
            return True
        else:
            return False

    def take_action(self, action):
        if type(action) == torch.Tensor:
            action = self.action_space[action.data.cpu().numpy()[0]]
        else:
            action = self.action_space[action]

        x, y = self.current_bin.coords
        new_x, new_y = x, y

        if action == 'UP': new_x = x - 1
        elif action == 'DOWN': new_x = x + 1
        elif action == 'LEFT': new_y = y - 1
        elif action == 'RIGHT': new_y = y + 1
        elif action == 'NOP':
            if self.current_bin.contains_goal_state:
                self.current_reward = self.reward_dict['goal_correct']
            else:
                self.current_reward = self.reward_dict['goal_incorrect']
            self.reset_grid()
            return torch.tensor([self.current_reward], device=self.device)

        if not self.check_boundaries(np.array([new_x, new_y])):
            self.current_reward = self.reward_dict['border_collision']
            if not self.plotting:
                self.border_collisions += 1
    #        self.set_grid(np.array(self.current_bin.coords))
            self.reset_grid()
        else:
            new_bin = self.bin_array[new_x, new_y]
            if self.is_closer(new_bin):
                self.current_reward = self.reward_dict['closer']
            else:
                self.current_reward = self.reward_dict['further']
            self.set_grid(np.array([new_x, new_y]))
        return torch.tensor([self.current_reward], device = self.device)

    def get_reward(self):
        return self.current_reward

    def load_frame(self, idx=False):
        if idx:
            frame = Image.open(self.frame_path + "sacrum_translation{:04d}.png".format(idx))
        #    frame = Image.open(self.frame_path + "frame_{:04d}.png".format(idx))
        else:
            new_frame_idx = random.randint(0,4)
            frame = Image.open(self.frame_path + "sacrum_translation{:04d}.png".format(self.current_bin.frame_idx[new_frame_idx]))
        #    frame = Image.open(self.frame_path + "frame_{:04d}.png".format(self.current_bin.frame_idx[new_frame_idx]))
        return np.array(frame)

    def get_frame_dims(self):
        frame = self.get_processed_frame()
        return frame.shape

    def get_state(self, frame_pos=None):
        if frame_pos:
            state = self.current_bin.frames[frame_pos, :, :]
            return state.unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            new_frame_idx = random.randint(0, 4)
            state = self.current_bin.frames[new_frame_idx, :, :]
            return state.unsqueeze(0).unsqueeze(0).to(self.device)
        #return self.get_processed_frame()

    def get_processed_frame(self, idx=False):
        frame = self.load_frame(idx)
        frame = np.ascontiguousarray(frame, dtype=np.float32) / 255
        frame = torch.from_numpy(frame)
        resize_x = 102
        resize_y = 108
        if self.cluster:
            resize_x = 272
            resize_y = 258
        # processed_frame = frame.view((1,1,frame.shape[0], -1)).to(self.device)
        resize = T.Compose([
            T.ToPILImage(),
            T.Resize((resize_x, resize_y)),
            T.ToTensor()
        ])
        base_frame = resize(frame).squeeze().numpy()
        frame_eq = np.sort(base_frame.ravel()).searchsorted(base_frame)

        bw_mask = frame_eq > 0
        disc = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0]])

        bw_mask = scipy.ndimage.binary_opening(bw_mask, disc)
        bw_mask = scipy.ndimage.binary_closing(bw_mask, disc)
        _, dy_frame = np.gradient(base_frame)

        frame = base_frame + dy_frame
        frame = frame * bw_mask
        frame = scipy.signal.medfilt(frame, 5)
        processed_frame = (frame - np.min(frame)) / np.ptp(frame)
        processed_frame = T.functional.to_tensor(processed_frame).unsqueeze(0).to(self.device)
        return processed_frame

    def display_frame(self, frame=None):
        if frame:
            frame = Image.open(self.frame_path + "sacrum_translation{:04d}.png".format(frame))
        else:
            frame = Image.open(self.frame_path + "sacrum_translation{:04d}.png".format(self.current_bin.frame_idx[0]))
        frame.show(title="Frame {}".format(self.current_bin.frame_idx[0]))
        return frame

    def is_closer(self, new_bin):
        return True if new_bin.distance(self.goal_state) < self.distance else False

    def num_actions_available(self):
        return len(self.action_space)

    def __str__(self):
        row = ""
        for i in range(self.bin_array.shape[0]):
            for j in range(self.bin_array.shape[1]):
                bin = self.bin_array[i, j]
                row += "{:4d} ".format(bin.frame_idx[0])
            print(row)
            row = ""
        return ""


def load_patient(file_path, data_path, args, cluster=False):
    with open(file_path) as file:
        name = file.readline().strip()
        _ = file.readline()
        frame_path = data_path + "Sacrum_{}/sacrum_sweep_frames/".format(name)
        print(frame_path)
        col_len = file.readline().strip().split(":")[1]
        goal_coords = file.readline().strip().split(":")[1:]
        goal_x, goal_y = 0, 0

        for i in range(len(goal_coords)):
            x, y = map(int, goal_coords[i].split(","))
            print(x, y)
            goal_x += x / len(goal_coords)
            goal_y += y / len(goal_coords)

        bin_grid = np.zeros((args.steps_x, args.steps_y), dtype=Bin)
        goals = []
        for _ in range(args.steps_x * args.steps_y):
            coords, frames = file.readline().strip().split(":")
            coord_x, coord_y = map(int, coords.split(","))
            bin = Bin(np.array([coord_x, coord_y]))
            if "{},{}".format(coord_x, coord_y) in goal_coords:
                bin.contains_goal_state = True
                goals.append(np.array([coord_x, coord_y]))

            frames = list(map(int, frames.split(",")))
            for i in range(5):
                bin.frame_idx.append(frames[i])
            bin_grid[coord_x, coord_y] = bin

        patient = Patient(name, bin_grid, frame_path, col_len, np.array([goal_x, goal_y]), args, cluster)
        patient.grid.goal_coords = goals

        return patient


def find_closest_frame(coords, data):  # Look into KDTrees!
    distance = (data[:, 1] - coords[0]) ** 2 + (data[:, 2] - coords[1]) ** 2
    frame_idxs = np.argsort(distance)
    frame_idx = data[frame_idxs[0], 0]
    #data[frame_idx, :] = 9999
    return int(frame_idx)

