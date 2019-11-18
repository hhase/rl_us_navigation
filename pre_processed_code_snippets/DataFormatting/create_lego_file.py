import os
import numpy as np
import utils.utility_functions as fun


def find_closest_frame(coords, data):  # Look into KDTrees!
    distance = (data[:, 1] - coords[0]) ** 2 + (data[:, 2] - coords[1]) ** 2
    frame_idxs = np.argsort(distance)
    frame_idx = data[frame_idxs[:5], 0]
    return frame_idx.astype(int)


name = "Lego_Phantom"

patient_path = "./../../../Data/Experiment_1/"
frame_path = patient_path + "frames/"

file_path = "./../../../Data/Experiment_1/lego_phantom_file.txt"

patient_file = open(file_path, "w")

patient_file.write(name + "\n")
patient_file.write("./../Data/Experiment_1/frames/\n")

steps_x = 5
steps_y = 20
margins = np.array([5,8])

header, data = fun.load_data(patient_path + "exp1_track_data.csv")
print(data.shape)
if data.shape[0] > 3000:
    col_size = 350
else:
    col_size = 250

data = fun.force_into_grid(data, cols=5, col_range=col_size)

patient_file.write("col_len:{}\n".format(col_size))

step_size = fun.def_step_size(steps_x, steps_y, margins, data)
print(step_size)

patient_file.write("goal_bins:goal_bin_x,goal_bin_y\n")

grid = np.zeros((steps_x, steps_y)).astype(int)
data_y_shift = np.nanmin(data[:, 2])
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        x_location = i * step_size[0] + margins[0]
        y_location = j * step_size[1] + margins[1]
        location = np.array([x_location, y_location + data_y_shift])
        print(location)
        coords = np.array([i, j])
        frame_idxs = find_closest_frame(location, data)
        grid[i,j] = frame_idxs[0]
        patient_file.write("{},{}:{}\n".format(i, j, ",".join(map(str, frame_idxs))))

patient_file.write("\n\n******************************************\n\n")

row = ""
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        frame = grid[i, j]
        row += "{:4d} ".format(frame)
    patient_file.write(row + "\n")
    print(row)
    row = ""

patient_file.close()
