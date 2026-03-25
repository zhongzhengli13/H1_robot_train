import numpy as np
from math import pi


class SetDict2Class:
    def set_dict(self, dict):
        for key, value in dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class H1Config:
    class viewer:
        pos = [3, 3, 2.0]  # [m]
        lookat = [0.0, 0, 1.5]  # [m]
        fixed_robot_id = 0
        fixed_offset = [0.0, 3.0, 1.6]

    class runner(SetDict2Class):
        seed = 1
        max_iterations = 4000  # number of policy updates
        num_steps_per_env = 24  # 24  # per iteration #该参数乘上 num_envs就是轨迹数量
        save_interval = 200  # check for potential saves every this many iterations
        epoch = 0

    class env(SetDict2Class):
        cfg = "loc"
        task = "LocomotionTask"
        num_envs = 4096
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 10  # episode length in seconds

    class policy(SetDict2Class):
        name = "simple_policy"
        num_actions = None
        num_critic_obs = None
        num_observations = None
        hidden_layers = (512, 256)  # (512, 256)  # #(128, 64, 32)
        activation = "relu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        eps_clip = 0.2
        entropy_coef = 0.001  # 会影响策略的收敛
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*n_steps / n_minibatches #
        learning_rate = 1e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        discount_factor = 0.993
        gae_lambda = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class action(SetDict2Class):
        """
        整体流程：
        1 策略网络输出 action
        2 action 被 clip 到 inc_low / inc_high
        3 累加到参考姿态：target_pos = ref_joint_pos + increment
        4 再被 clip 到 low_ranges / high_ranges
        5 送入 PD 控制器
        """

        action_limit_up = None
        action_limit_low = None
        """
        high_ranges / low_ranges -> 这是 绝对动作范围（通常用于：use_increment = False 或作为最终 safety clip）
        可以理解为：“无论你怎么加，都不能超过这个关节安全范围”
        # """
        # high_ranges = [3.5] * 4 + [1.0, 1.8, -0.5] * 4
        # low_ranges = [0.5] * 4 + [-1.0, -0.2, -2.2] * 4
        # 修改
        # HipYaw, HipRoll, HipPitch, Knee, Ankle (左右腿)
        # 原始
        # high_ranges = [1.0, 1.0, 3.0, 2.6, 1.0] * 2
        # low_ranges = [-1.0, -1.0, -3.0, -0.5, -1.0] * 2
        # 修改 gem
        # high_ranges = [0.5, 0.5, 2.6, 2.1, 0.6] * 2
        # low_ranges = [-0.5, -0.5, -3.14, -0.3, -0.9] * 2
        high_ranges = [0.43, 0.43, 2.53, 2.05, 0.52] * 2
        low_ranges = [-0.43, -0.43, -3.14, -0.26, -0.87] * 2

        ref_joint_pos = (
            [0.0, 0.0, -0.2, 0.4, -0.2] * 2 + [0] + ([0.0] * 4) * 2
        )  # 这是一整个人形机器人的默认站姿

        # 策略不是直接给“关节目标角度”，而是给“在当前基础上，加多少”。即为：新目标 = 参考姿态 + 累积增量。相比较比 直接控制绝对角度 稳定得多。
        use_increment = True
        """
        inc_high_ranges / inc_low_ranges --》 每一维 action，最多能“加 / 减”多少
        公式中的前 2 个（通常是 base / 全局相关），后 10 个（关节）
        """
        # inc_high_ranges = [3.5] * 2 + [
        #     12.0
        # ] * 10  # + [12.] + [5.] * 8  # f*2+ (hip_yaw, hip_roll, hip_pith, knee_pitch, ankle_pith,should roll,shoulder pitch,elbow) * 2
        # inc_low_ranges = [0.5] * 2 + [-12.0] * 10  # + [-12.] + [-5.] * 8
        # 增量范围可以保持较大的值，或者稍微收紧一点以防动作太猛 修改 原始
        # inc_high_ranges = [2.0] * 2 + [5.0] * 10
        # inc_low_ranges = [-2.0] * 2 + [-5.0] * 10
        # 修改 gem
        # inc_high_ranges = [0.1, 0.1] + [0.3] * 10
        # inc_low_ranges = [-0.1, -0.1] + [-0.3] * 10
        inc_high_ranges = [3.5, 3.5] + [12.0] * 10
        inc_low_ranges = [0.5, 0.5] + [-12.0] * 10

    class pd_gains(SetDict2Class):
        decimation = 10
        # stiffness = {
        #     "hip_yaw": 200,
        #     "hip_roll": 200,
        #     "hip_pitch": 200,
        #     "knee": 300,
        #     "ankle": 40,
        #     "torso": 300,
        #     "shoulder": 200,
        #     "elbow": 200,
        # }  # [N*m/rad] #原始
        stiffness = {
            "hip_yaw": 290,
            "hip_roll": 290,
            "hip_pitch": 250,
            "knee": 290,
            "ankle": 270,
            "torso": 300,
            "shoulder": 200,
            "elbow": 200,
        }
        # damping = {
        #     "hip_yaw": 15.0,
        #     "hip_roll": 30.0,
        #     "hip_pitch": 15.0,
        #     "knee": 25,
        #     "ankle": 4.0,
        #     "torso": 10,
        #     "shoulder": 5,
        #     "elbow": 5,
        # }  # [N*m*s/rad] #原始
        damping = {
            "hip_yaw": 8.0,
            "hip_roll": 8.0,
            "hip_pitch": 8.0,
            "knee": 9.0,
            "ankle": 7.0,
            "torso": 2.0,
            "shoulder": 2.0,
            "elbow": 2.0,
        }

    class init_state(SetDict2Class):  # 初始状态
        random_rot = True
        num_legs = 2
        pos = [0.0, 0.0, 1.06]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0] * 3  # x,y,z [m/s]
        ang_vel = [0.0] * 3  # x,y,z [rad/s]
        reset_joint_pos = [0.0, 0.0, -0.2, 0.4, -0.2] * \
            2 + [0] + ([0.0] * 4) * 2
        # reset_joint_pos = [0.] * 19

    class domain_rand(SetDict2Class):  # 域随机化
        randomize_friction = False
        friction_range = [0.65, 1.0]
        randomize_mass = False  # muti
        added_mass_range = [0.9, 1.2]
        added_inertia_range = [0.9, 1.2]
        randomize_mass_com = False  # add
        added_mass_com_high = [0.01, 0.01, 0.01]
        added_mass_com_low = [-0.01, -0.01, -0.01]
        added_body_mass = False  # add 17.789kg
        added_body_mass_range = [-0.5, 0.5]
        added_body_inertia_range = [0.95, 1.55]
        randomize_body_com = False  # add
        added_body_mass_com_high = [0.01, 0.01, 0.01]
        added_body_mass_com_low = [-0.01, -0.01, -0.01]
        randomize_damping = False
        added_damping_range = [0.8, 1.2]
        added_friction_range = [0.8, 1.2]
        randomize_torque = False
        torque_range = [0.95, 1.05]
        randomize_gains = False
        gains_range = [0.9, 1.3]
        push_robots = False
        push_interval_s = 3.0
        max_push_force = 100
        push_duration_step = 200
        delay_observation = False
        delay_joint_ranges = [10, 20]  # [4, 10]  # 5~16 ms for q, dq
        # [50, 100]  # 10~40 ms for angle velocity
        delay_rate_ranges = [10, 20]
        delay_angle_ranges = [
            5,
            10,
        ]  # [80, 150]  # 10~40 ms for base euler and base linear velocity
        randomize_joint_static_error = False
        added_joint_static_error = 0.01

    class noise_values(SetDict2Class):
        randomize_noise = False
        # lin_vel = 0.15
        gravity = 0.3
        ang_vel = 0.2
        # foot_frc = 5.
        dof_pos = 0.01  # 编码器位置噪声（rad）
        dof_vel = 0.1  # 由“差分”算出来的速度噪声（rad/s）

    class command(SetDict2Class):
        """
        1️⃣ command = “走路目标”
        2️⃣ resampling_time = 多久换一次目标
        3️⃣ *_range = 这个目标最多给多大
        """

        curriculum = False
        max_curriculum = 1.0
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading, 4target_foot_height (in heading mode ang_vel_yaw is recomputed from heading error) #每一条指令由 4 个数字组成
        num_commands = 4
        resampling_time = (
            4.0  # time before command are changed[s] #每 4 秒，换一次‘目标指令’
        )
        # if true: compute ang vel command from heading error #False:直接告诉你转多快;Ture:只告诉你“面朝哪”
        heading_command = False
        # 修改
        lin_vel_x_range = [0.3, 0.7]
        # lin_vel_x_range = [-0.1, 0.1]  # 最多向前：0.1 m/s;最多向后：0.1 m/s #原始
        # 修改
        ang_vel_yaw_range = [0, 0]
        # ang_vel_yaw_range = [-0.1, 0.1]  # 最多慢慢左转;最多慢慢右转
        lin_vel_y_range = [0.0, 0.0]  # 不允许横着走
        heading_range = [0, 0]  # heading_command = True 才生效——》 “给一个很小的目标朝向变化”

    class terrain(SetDict2Class):
        mesh_type = "plane"  # none, plane, heightfield or trimesh #地形
        # [m] #高度图中一个格子在水平方向对应现实多少米 #在现实中 = 0.1 米（10 cm）(正方形的格子)
        horizontal_scale = 0.1
        vertical_scale = (
            0.01  # [m] #高度图中“高度值 1”代表多少米 #实际高度 = 10 × 0.01 = 0.1 m
        )
        border_size = 5  # [m]
        static_friction = 1.0  # 静摩擦系数（越大越粗糙）
        dynamic_friction = 1.0  # 动摩擦系数，通常 静摩擦 ≥ 动摩擦
        restitution = (
            0.5  # 反弹系数（弹性）范围：[0，1] 其中 0为完全不反弹，1为完全弹回
        )
        """
        horizontal_scale 决定地形“每一格有多大”；
        measured_points_x / y 决定机器人“在脚下哪些位置量高度”。
        它们是同一坐标系下的尺度关系。
        举例子：地面是用 10 厘米的瓷砖铺的，机器人每走一步，会在脚下这块瓷砖的中间、前后左右用手摸一摸地面是不是平。
        """
        # rough terrain only:
        measured_points_x = [-0.05, 0, 0.05]  # 前后
        measured_points_y = [-0.05, 0, 0.05]  # 左右
        curriculum = False  # 课程学习：从简单到难
        measure_heights = False
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 3  # starting curriculum state
        terrain_length = 8
        terrain_width = 8
        num_rows = 20  # number of terrain rows (levels)  15
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.2, 0.3, 0.2, 0.2]
        # trimesh only:
        slope_treshold = (
            0.0  # slopes above this threshold will be corrected to vertical surfaces
        )

    class sim:
        dt = 0.001  # 0.001 #配合decimation，两者相乘才是策略执行的时间 10*0.001=0.01s
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]

        class physx:
            solver_type = 1  # 0: pgs, 1: tgs
            num_threads = 10
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = (
                2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            )
            use_gpu = True

    class asset:
        enable_bar = False
        file = "assets/h1/urdf/h1.urdf"
        foot_name = ["ankle"]
        penalize_contacts_on = ["ankle"]
        shoulder_roll_name = ["shoulder_roll"]
        terminate_after_contacts_on = [
            "elbow",
            "yaw",
            "roll",
            "pitch",
            "torso",
            "knee",
            "hip",
        ]
        disable_gravity = False
        # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        collapse_fixed_joints = True
        fix_base_link = False  # fixe the base of the robot == on rack
        # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        default_dof_drive_mode = 3
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        # replace collision cylinders with capsules, leads to faster/more stable simulation
        replace_cylinder_with_capsule = True
        flip_visual_attachments = (
            False  # Some .obj meshes must be flipped from y-up to z-up
        )
        use_mesh_materials = True  # color!!! False have color

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 100.0
        max_linear_velocity = 100.0
        armature = 0.0
        thickness = 0.01
