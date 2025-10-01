import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_file = "state_estimation_log.csv"

# Wait for the log file to appear (optional)
wait_time = 0
while not os.path.exists(csv_file) and wait_time < 30:
    print(f"Waiting for {csv_file} to be created...")
    time.sleep(1)
    wait_time += 1

if not os.path.exists(csv_file):
    print(f"File {csv_file} not found. Exiting.")
    exit(1)

# Read the CSV log
df = pd.read_csv(csv_file)

# time_list = np.linspace(0, 10, len(df['xy_vel_x']))
dt = 0.005
time_list = df.index * dt
# Plot
plt.figure(figsize=(10, 6))
# plt.plot(df['time'][:250], df['ang_vel_x'][:250], label='angular vel x', linewidth=2)
# plt.plot(df['time'], df['ang_vel_y'], label='angular vel y', linewidth=2)
# plt.plot(df['time'][:250], df['ang_vel_z'][:250], label='angular vel z', linewidth=2)
# plt.plot(df['time'], df['project_gravity_x'], label='project gravity x', linewidth=2)
# plt.plot(df['time'][:250], df['project_gravity_y'][:250], label='project gravity y', linewidth=2)
# plt.plot(df['time'][:250], df['project_gravity_z'][:250], label='project gravity z', linewidth=2)

# plt.plot(time_list, df['ang_vel_x'], label='angular vel x')
# plt.plot(time_list, df['ang_vel_y'], label='angular vel y')
# plt.plot(time_list, df['ang_vel_z'], label='angular vel z')
# plt.scatter(time_list, df['gravity_x'], label='project gravity x')
# plt.scatter(time_list, df['gravity_y'], label='project gravity y')
# plt.scatter(time_list, df['gravity_z'], label='project gravity z')

# plt.plot(df['time'], df['l_wheel_cmd'], label='l wheel cmd', linewidth=2, linestyle='--')
# plt.plot(time_list, df['qtau_l_t'], label='l thigh tau')
# plt.plot(time_list, df['qtau_l_c'], label='l calf tau')
# plt.plot(time_list, df['qtau_l_w'], label='l wheel tau')
# plt.plot(time_list, df['qtau_r_t'], label='r thigh tau')
# plt.plot(time_list, df['qtau_r_c'], label='r calf tau')
# plt.plot(time_list, df['qtau_r_w'], label='r wheel tau')

# plt.plot(time_list, df['action_l_t'], label='l thigh cmd')
# plt.plot(time_list, df['action_l_c'], label='l calf cmd')
# plt.plot(time_list, df['action_l_w'], label='l wheel cmd')
# plt.plot(time_list, df['action_r_t'], label='r thigh cmd')
# plt.plot(time_list, df['action_r_c'], label='r calf cmd')
# plt.plot(time_list, df['action_r_w'], label='r wheel cmd')
# plt.plot(time_list, df['z_est'], label='base height')
plt.plot(time_list, df['xy_vel_x'], label='linear vel x')
# plt.plot(time_list, df['xy_vel_y'], label='linear vel y')
plt.plot(time_list, df['cmd_x'], label='cmd x', linestyle='--')
# plt.plot(time_list, df['cmd_yaw'], label='cmd yaw', linestyle='--')
# plt.plot(time_list, df['ang_vel_x'], label='angular vel x')
# plt.plot(time_list, df['ang_vel_y'], label='angular vel y')
# plt.plot(time_list, df['ang_vel_z'], label='angular vel z')
# plt.plot(time_list, df['lin_vel_cmd'], label='lin vel cmd')
# plt.plot(time_list, df['ang_vel_cmd'], label='ang vel cmd')

plt.xlabel('Time (s)')
plt.ylabel('')
plt.title('Obs state')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("state_estimation.png")
plt.show()
