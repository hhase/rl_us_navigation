from transforms3d import euler

mat_path = "./../../../Data/Sacrum_Zhonliang/sacrum_sweep_mat.txt"
csv_path = "./../../../Data/Sacrum_Zhonliang/sacrum_sweep_data.csv"

header = "frame,dx,dy,dz,e1,e2,e3,r11,r12,r13,r21,r22,r23,r31,r32,r33\n"
formatted_line = ""
mat_content = ["asd"]

with open(mat_path, 'r') as mat_file:
    with open(csv_path, 'w') as csv_file:
        csv_file.write(header)
        while(True):
            mat_content = []
            for i in range(5):
                mat_content.append(mat_file.readline().strip())

            if not mat_content:
                break

            print(mat_content[0].strip())

            frame = mat_content[0].strip().split("_")[1]
            r11 = mat_content[1].strip().split()[0]
            r21 = mat_content[1].strip().split()[1]
            r31 = mat_content[1].strip().split()[2]
            dx  = mat_content[1].strip().split()[3]
            r12 = mat_content[2].strip().split()[0]
            r22 = mat_content[2].strip().split()[1]
            r32 = mat_content[2].strip().split()[2]
            dy  = mat_content[2].strip().split()[3]
            r13 = mat_content[3].strip().split()[0]
            r23 = mat_content[3].strip().split()[1]
            r33 = mat_content[3].strip().split()[2]
            dz  = mat_content[3].strip().split()[3]

            e1, e2, e3 = euler.mat2euler([[r11, r12, r13], [r21, r22, r23],[r31, r32, r33]])

            csv_file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(frame, dx, dy, dz, e1, e2, e3, r11, r12, r13, r21, r22, r23, r31, r32, r33))
