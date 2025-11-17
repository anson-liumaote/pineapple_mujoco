import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import argparse
import threading
import msvcrt  # Windows 鍵盤即時輸入

# from gamepaded import gamepad_reader
# NUM_MOTOR = 6 --- IGNORE ---

def calculate_com_in_base_frame(model, data, base_body_id):
    total_mass = 0.0
    com_sum = np.zeros(3)

    # Get base position and rotation
    base_pos = data.xipos[base_body_id]  # Position of the base in world coordinates
    base_rot = data.ximat[base_body_id].reshape(3, 3)  # Rotation matrix of the base

    for i in range(model.nbody):
        # Get body mass and world COM position
        mass = model.body_mass[i]
        world_com = data.xipos[i]

        # Transform COM to base coordinates
        local_com = world_com - base_pos  # Translate to base origin
        local_com = base_rot.T @ local_com  # Rotate into base frame

        # Accumulate mass-weighted positions
        com_sum += mass * local_com
        total_mass += mass

    # Compute overall COM in base coordinates
    center_of_mass_base = com_sum / total_mass
    return center_of_mass_base

def quat_rotate_inverse(q, v):
    """
    Rotate a vector by the inverse of a quaternion.
    Direct translation from the PyTorch version to NumPy.
    
    Args:
        q: The quaternion in (w, x, y, z) format. Shape is (..., 4).
        v: The vector in (x, y, z) format. Shape is (..., 3).
        
    Returns:
        The rotated vector in (x, y, z) format. Shape is (..., 3).
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    
    # Equivalent to (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    term1 = 2.0 * np.square(q_w) - 1.0
    term1_expanded = np.expand_dims(term1, axis=-1)
    a = v * term1_expanded
    
    # Equivalent to torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    q_w_expanded = np.expand_dims(q_w, axis=-1)
    b = np.cross(q_vec, v) * q_w_expanded * 2.0
    
    # Equivalent to the torch.bmm or torch.einsum operations
    # This calculates the dot product between q_vec and v
    dot_product = np.sum(q_vec * v, axis=-1)
    dot_product_expanded = np.expand_dims(dot_product, axis=-1)
    c = q_vec * dot_product_expanded * 2.0
    
    return a - b + c

def get_gravity_orientation(quaternion):
    """
    Get the gravity vector in the robot's base frame.
    Uses the exact algorithm from your PyTorch code.
    
    Args:
        quaternion: Quaternion in (w, x, y, z) format.
        
    Returns:
        3D gravity vector in the robot's base frame.
    """
    # # Ensure quaternion is a numpy array
    # quaternion = np.array(quaternion)
    
    # # Standard gravity vector in world frame (pointing down)
    # gravity_world = np.array([0, 0, -1])
    
    # # Handle both single quaternion and batched quaternions
    # if quaternion.shape == (4,):
    #     quaternion = quaternion.reshape(1, 4)
    #     gravity_world = gravity_world.reshape(1, 3)
    #     result = quat_rotate_inverse(quaternion, gravity_world)[0]
    # else:
    #     gravity_world = np.broadcast_to(gravity_world, quaternion.shape[:-1] + (3,))
    #     result = quat_rotate_inverse(quaternion, gravity_world)
    q = np.array(quaternion)
    gravity = np.zeros(3, dtype=np.float32)
    gravity[0]=2*(-q[1]*q[3]+q[0]*q[2])
    gravity[1]=-2*(q[2]*q[3]+q[0]*q[1])
    gravity[2]=1-2*(q[0]*q[0]+q[3]*q[3])
    return gravity
    # return result

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        policy = torch.jit.load(policy_path)
        xml_path = config["xml_path"]
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        pos_action_scale = config["pos_action_scale"]
        vel_action_scale = config["vel_action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        one_step_obs_size = config["one_step_obs_size"]
        obs_buffer_size = config["obs_buffer_size"]

        leg_joint_indices = config["leg_joint_indices"]
        wheel_joint_indices = config["wheel_joint_indices"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # 新增：鍵盤控制參數與共享指令向量 (假設 cmd_vel = [vx, vy, yaw])
    cmd_vel = np.array(config["cmd_init"], dtype=np.float32)
    lin_step = 0.1
    ang_step = 0.1
    max_lin = 2.0
    max_ang = 3.0
    cmd_lock = threading.Lock()
    stop_event = threading.Event()

    def keyboard_thread():
        global cmd_vel
        print("[Keyboard]: W/S: linear velocity, A/D: angular velocity, Space: stop, Esc: exit")
        while not stop_event.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                with cmd_lock:
                    if ch in ('w', 'W'):
                        cmd_vel[0] = np.clip(cmd_vel[0] + lin_step, -max_lin, max_lin)
                    elif ch in ('s', 'S'):
                        cmd_vel[0] = np.clip(cmd_vel[0] - lin_step, -max_lin, max_lin)
                    elif ch in ('a', 'A'):
                        cmd_vel[2] = np.clip(cmd_vel[2] + lin_step, -max_lin, max_ang)
                    elif ch in ('d', 'D'):
                        cmd_vel[2] = np.clip(cmd_vel[2] - lin_step, -max_lin, max_ang)
                    elif ch == ' ':
                        cmd_vel[:] = 0
                    elif ord(ch) == 27:  # Esc
                        stop_event.set()
                        break
                    print(f"[Keyboard] cmd_vel = {cmd_vel}")
            time.sleep(0.01)

    kb_thread = threading.Thread(target=keyboard_thread, daemon=True)
    kb_thread.start()

    target_dof_pos = default_angles.copy()
    target_dof_vel = np.zeros(num_actions)
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)
    obs_tensor_buf = torch.zeros((1, one_step_obs_size * obs_buffer_size))

    # gamepad = gamepad_reader.Gamepad(vel_scale_x=0.5, vel_scale_y=0.5, vel_scale_rot=0.8)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    base_body_id = 1


    # Record data
    lin_vel_data_list = []
    ang_vel_data_list = []
    gravity_b_list = []
    joint_pos_list = []
    joint_vel_list = []
    action_list = []
    time_list = []
    cmd_list = []
    tau_list = []
    # time_list = [i * simulation_dt for i in range(int(simulation_duration / simulation_dt))]

    counter = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        mujoco.mj_resetDataKeyframe(m, d, 0)
        viewer.sync()
        while viewer.is_running() and counter*simulation_dt < simulation_duration:
            step_start = time.time()
    
            tau = pd_control(target_dof_pos, d.sensordata[:num_actions], kps, target_dof_vel, d.sensordata[num_actions:num_actions + num_actions], kds)
            
            # if time.time() - start > 3:
            d.ctrl[:] = tau

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            # com_base = calculate_com_in_base_frame(m, d, base_body_id)
            # print("Center of Mass in Base Coordinates:", com_base)

            # After mujoco.mj_step(m, d)
            robot_pos = d.xipos[base_body_id]  # Get robot base position in world coordinates
            viewer.cam.lookat[:] = robot_pos   # Set camera to look at robot base
            # viewer.cam.distance = 3.0          # Comment out to allow mouse zoom
            # viewer.cam.elevation = -20         # Comment out to allow mouse pitch
            # viewer.cam.azimuth = 90            # Comment out to allow mouse yaw

            counter += 1
            # qvel = d.sensordata[6:12]
            # joint_vel_list.append(qvel.copy())
            # action_list.append(target_dof_vel.copy())
            if counter % control_decimation == 0 and counter > 0:

                # create observation
                qpos = d.sensordata[:num_actions]
                qvel = d.sensordata[num_actions: 2*num_actions]
                imu_quat = d.sensordata[3*num_actions:3*num_actions+4]
                ang_vel_I = d.sensordata[3*num_actions+4:3*num_actions+4+3]
                
                lin_vel_I = d.sensordata[3*num_actions+4+3+6:3*num_actions+4+3+6+3]

                # 取用即時鍵盤速度指令
                with cmd_lock:
                    current_cmd_vel = cmd_vel.copy()

                ang_vel_B = quat_rotate_inverse(imu_quat, ang_vel_I)
                lin_vel_B = quat_rotate_inverse(imu_quat, lin_vel_I)
                gravity_b = get_gravity_orientation(imu_quat)

                # SAFE leg joint delta (1D)
                valid_leg_idx = [i for i in leg_joint_indices if i < len(qpos) and i < len(default_angles)]
                leg_pos_delta = (qpos[valid_leg_idx] - default_angles[valid_leg_idx]) * dof_pos_scale
                leg_pos_delta = leg_pos_delta.astype(np.float32).ravel()
                
                obs_list = [
                    current_cmd_vel * cmd_scale,
                    ang_vel_I * ang_vel_scale,
                    gravity_b,
                    leg_pos_delta,
                    qvel * dof_vel_scale,
                    action.astype(np.float32)
                ]
                ## Record Data ##
                lin_vel_data_list.append(lin_vel_B.copy())
                ang_vel_data_list.append(ang_vel_I.copy())
                gravity_b_list.append(gravity_b)
                joint_pos_list.append(qpos.copy())
                joint_vel_list.append(qvel.copy())
                # action_list.append(action*pos_action_scale+default_angles)
                action_list.append(action*vel_action_scale)
                time_list.append(counter * simulation_dt)
                cmd_list.append(current_cmd_vel.copy()) 
                tau_list.append(tau.copy())
                ###
                obs_list = [torch.tensor(obs, dtype=torch.float32) if isinstance(obs, np.ndarray) else obs for obs in obs_list]

                obs = torch.cat(obs_list, dim=0).unsqueeze(0)

                # print("obs_list shapes:", [o.shape for o in obs_list])
                # print("obs shape:", obs.shape)

                obs_tensor_buf = torch.cat([
                    obs,
                    obs_tensor_buf[:, :-one_step_obs_size]
                ], dim=1)

                obs_tensor_buf = torch.clip(obs_tensor_buf, -100, 100)
                # policy inference
                action = policy(obs_tensor_buf).detach().numpy().squeeze()
                # print("action :", action)

                # Set leg joint target positions
                for idx in leg_joint_indices:
                    if idx < len(target_dof_pos) and idx < len(action):
                        target_dof_pos[idx] = default_angles[idx] + action[idx] * pos_action_scale
                        # target_dof_pos[idx] = default_angles[idx]
                # Set wheel joint target velocities
                for idx in wheel_joint_indices:
                    if idx < len(target_dof_vel) and idx < len(action):
                        target_dof_vel[idx] = action[idx] * vel_action_scale

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    # Plot the collected data after the simulation ends
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    for i in range(3): 
        plt.plot(time_list, [step[i] for step in lin_vel_data_list], label=f"Linear Velocity {i}")
    plt.plot(time_list, [step[0] for step in cmd_list], label=f"Command Velocity x", linestyle='--')
    plt.title(f"History Linear Velocity", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.grid()
    plt.subplot(2, 2, 2)
    for i in range(3):
        plt.plot(time_list, [step[i] for step in ang_vel_data_list], label=f"Angular Velocity {i}")
    plt.plot(time_list, [step[2] for step in cmd_list], label=f"Command Velocity yaw", linestyle='--')
    plt.title(f"History Angular Velocity", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.grid()
    plt.subplot(2, 2, 3)
    for i in range(3):
        plt.plot(time_list, [step[i] for step in gravity_b_list], label=f"Project Gravity {i}")
    plt.title(f"History Project Gravity", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.grid()
    plt.subplot(2, 2, 4)
    for i in [3,7]:
        plt.plot(time_list, [step[i] for step in action_list], label=f"Joint action {i}", linestyle='--')
        plt.plot(time_list, [step[i] for step in joint_vel_list], label=f"Joint vel {i}")
    for i in range(3):
        plt.plot(time_list, [step[i] for step in ang_vel_data_list], label=f"Angular Velocity {i}")
    plt.title(f"History Joint Action-Position", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("history_data.png", dpi=300)
    # # plt.show()

    stop_event.set()
    kb_thread.join(timeout=1.0)