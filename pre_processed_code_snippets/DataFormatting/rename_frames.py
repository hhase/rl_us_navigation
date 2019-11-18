import os

frame_path = "./../../../Data/Experiment_1/frames/"
name_format = "frame_{:04d}.png"

for file_name in os.listdir(frame_path):
    #idx = file_name.split(" ")[1]
    #idx = idx.split(".")[0]
    #if idx[1:]:
    #    os.rename(frame_path + file_name, frame_path + name_format.format(int(idx[1:])))
    #else:
    #    os.rename(frame_path + file_name, frame_path + name_format.format(0))
    if "frame" not in file_name:
        os.rename(frame_path + file_name, frame_path + name_format.format(int(file_name.strip("_"))))
        print(name_format.format(int(file_name.strip("_"))))

