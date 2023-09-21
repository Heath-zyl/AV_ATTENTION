import csv
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm


filename = 'vehicle_tracks_000.csv'
sample = []
with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    data = list(reader)
    sorted_data = sorted(data, key=lambda x: int(x[1]))   # 按照frame_id排序
    last_ego_id = -1
    ego_path_dict = {}
    for row in tqdm(data):
        ego_id = eval(row[0])
        if last_ego_id != ego_id:
            # print(ego_id)
            ego_path_dict[row[0]] = []
        frame_id = eval(row[1])
        ego_x = eval(row[4])
        ego_y = eval(row[5])
        ego_vx = eval(row[6])
        ego_vy = eval(row[7])
        ego_psi_rad = eval(row[8])
        point = [frame_id, ego_x, ego_y, ego_psi_rad]
        ego_path_dict[row[0]].append(point)
        last_ego_id = ego_id
        ego_veh = [ego_id, ego_x, ego_y, ego_vx, ego_vy, ego_psi_rad]

        traffic_veh_list = []
        for i in range(len(sorted_data)):
            track_id = eval(sorted_data[i][0])
            tmp_frame_id = eval(sorted_data[i][1])
            if tmp_frame_id > frame_id:
                break
            if tmp_frame_id == frame_id and ego_id != track_id:
                traffic_id = track_id
                traffic_x = eval(sorted_data[i][4])
                traffic_y = eval(sorted_data[i][5])
                traffic_vx = eval(sorted_data[i][6])
                traffic_vy = eval(sorted_data[i][7])
                traffic_psi_rad = eval(sorted_data[i][8])
                traffic_veh = [traffic_id, traffic_x, traffic_y, traffic_vx, traffic_vy, traffic_psi_rad]
                traffic_veh_list.append(traffic_veh)
        tmp_sample = {'frame_id': frame_id, 'ego_veh': ego_veh, 'traffic_veh_list': traffic_veh_list}
        sample.append(tmp_sample)
train_sample = []
Tg = 1
for i in range(len(sample)):
    ego_id = sample[i]['ego_veh'][0]
    frame_id = sample[i]['frame_id']
    ego_path = ego_path_dict[str(ego_id)]
    ego_future_path = []
    for j in range(len(ego_path)):
        if ego_path[j][0] >= frame_id:
            ego_future_path.append((ego_path[j][1], ego_path[j][2], ego_path[j][3]))
    if len(ego_future_path) > Tg * 10:
        distances = [0]
        for k in range(1, len(ego_future_path)):
            x1, y1, yaw1 = ego_future_path[k - 1]
            x2, y2, yaw2 = ego_future_path[k]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distances.append(distances[-1] + distance)

        x = [point[0] for point in ego_future_path]
        y = [point[1] for point in ego_future_path]
        yaw = [point[1] for point in ego_future_path]

        # 创建插值函数
        f_x = interp1d(distances, x, kind='linear')
        f_y = interp1d(distances, y, kind='linear')
        f_yaw = interp1d(distances, yaw, kind='linear')

        # 定义新的路程值（等路程间隔）
        new_distances = np.arange(0, distances[-1], 0.2)  # 此处步长为1.0，可以根据需要调整

        # 使用插值函数计算对应的 x、y 值和航向角
        new_x = f_x(new_distances)
        new_y = f_y(new_distances)
        new_headings = f_yaw(new_distances)
        new_ego_future_path = list(zip(new_x, new_y, new_headings))

        sample[i]['ego_future_path'] = new_ego_future_path

        Se = distances[Tg*10]
        S0 = distances[0]
        ego_vx = sample[i]['ego_veh'][3]
        ego_vy = sample[i]['ego_veh'][4]
        ego_psi_rad = sample[i]['ego_veh'][5]
        V0 = ego_vx * np.cos(ego_psi_rad) + ego_vy * np.sin(ego_psi_rad)
        acc = 2 * (Se - S0 - V0 * Tg) / pow(Tg, 2)
        sample[i]['ego_action'] = acc
        train_sample.append(sample[i])
train_sample = np.array(train_sample)
np.save('train_sample.npy', train_sample)

