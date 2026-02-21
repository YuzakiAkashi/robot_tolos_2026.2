#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import random
import os
import math
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler

# ==============================================================================
# 全局配置区域 (Global Configuration)
# ==============================================================================

# --- A, B, C 系列模型配置  ---
MODEL_LIBRARY_A = ["A000", "A001", "A010", "A011", "A111"]
NUM_MODELS_A = 6
SPAWN_CONFIG_A = [
    {'x': -0.643526,  'y': 0.46,  'z': 0, 'Yaw_degrees': 90}, {'x': 0.218,  'y': 0.46, 'z': 0, 'Yaw_degrees': -90},
    {'x': -0.643526,  'y': 1.01,  'z': 0, 'Yaw_degrees': 90}, {'x': 0.218,  'y': 1.01, 'z': 0, 'Yaw_degrees': -90},
    {'x': -0.643526,  'y': 1.56,  'z': 0, 'Yaw_degrees': 90}, {'x': 0.218,  'y': 1.56, 'z': 0, 'Yaw_degrees': -90},
]

MODEL_LIBRARY_B = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]
NUM_MODELS_B = 3
SPAWN_CONFIG_B = [
    {'x': 0.82654, 'y': 0.26, 'z': 0, 'Yaw_degrees': 0}, 
    {'x': 0.82654, 'y': 1.01, 'z': 0, 'Yaw_degrees': 180},
    {'x': 0.82654, 'y': 1.76, 'z': 0, 'Yaw_degrees': 0},
]

MODEL_LIBRARY_C = ["C000", "C001", "C010", "C011", "C100", "C101", "C110", "C111"]
NUM_MODELS_C = 3
SPAWN_CONFIG_C = [
    {'x': 1.72, 'y': 0.46, 'z': 0, 'Yaw_degrees': -90}, 
    {'x': 1.72, 'y': 1.01, 'z': 0, 'Yaw_degrees': -90},
    {'x': 1.72, 'y': 1.56, 'z': 0, 'Yaw_degrees': -90},
]

# --- QR码系列模型配置  ---
MODEL_LIBRARY_D_QR = ["qr_1326", "qr_1623"] 
NUM_MODELS_D_QR = 1
SPAWN_CONFIG_D_QR = [
    {'x': 2.59, 'y': 4.63, 'z': 0, 'Yaw_degrees': 0},
]

# --- 锥桶系列程序化配置  ---
MODEL_LIBRARY_E_CONE = ["锥桶1", "锥桶2", "锥桶3",] 
NUM_MODELS_E_CONE = 6
MAX_GROUPS_CONE = 4
# 【核心参数】
MIN_GROUP_DISTANCE_X = 1.3
START_X = 5
#  锥桶与道路中心线(Y=0)的横向距离随机范围
LATERAL_OFFSET_Y_MIN = 1
LATERAL_OFFSET_Y_MAX = 2

# ==============================================================================

def delete_all_spawned_models():
    """删除之前由该脚本生成的所有系列模型"""
    rospy.loginfo("开始清理场景中的旧模型...")
    try:
        delete_model_proxy = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        
        model_configs = [
            ("random_A_model", NUM_MODELS_A), ("random_B_model", NUM_MODELS_B),
            ("random_C_model", NUM_MODELS_C), ("random_D_QR_model", NUM_MODELS_D_QR),
            ("random_E_CONE_model", NUM_MODELS_E_CONE)
        ]

        for prefix, num in model_configs:
            for i in range(num):
                model_name = f"{prefix}_{i}"
                rospy.loginfo(f"  - 尝试删除: {model_name}")
                delete_model_proxy(model_name)
    except rospy.ServiceException:
        rospy.loginfo("删除服务当前不可用或模型不存在，将继续。")


def spawn_model_from_library(library_model_name, gazebo_instance_name, pose):
    """从库中读取SDF并生成一个指定模型"""
    sdf_path = os.path.expanduser(f'~/.gazebo/models/{library_model_name}/model.sdf')
    try:
        with open(sdf_path, 'r') as sdf_file:
            sdf_string = sdf_file.read()
    except Exception as e:
        rospy.logerr(f"读取SDF文件 '{sdf_path}' 失败: {e}")
        return
    try:
        spawn_model_proxy = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_proxy(model_name=gazebo_instance_name, model_xml=sdf_string, robot_namespace="", initial_pose=pose, reference_frame="world")
        rospy.loginfo(f"成功生成模型 '{gazebo_instance_name}' (类型: {library_model_name})")
    except rospy.ServiceException as e:
        rospy.logerr(f"模型生成服务调用失败: {e}")

def process_and_spawn_models(series_name, num_models, library, config):
    """一个通用的函数，用于处理和生成一个系列的【固定或动态】配置模型"""
    rospy.loginfo(f"--- 开始生成 {num_models} 个{series_name}系列模型 ---")
    for i in range(num_models):
        chosen_model_type = random.choice(library)
        instance_name = f"random_{series_name}_model_{i}"
        pose_config = config[i]
        yaw_radians = math.radians(pose_config['Yaw_degrees'])
        q = quaternion_from_euler(0, 0, yaw_radians)
        spawn_pose = Pose(position=Point(x=pose_config['x'], y=pose_config['y'], z=pose_config['z']), orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))
        spawn_model_from_library(chosen_model_type, instance_name, spawn_pose)
        rospy.sleep(0.2)

# --- 程序化生成锥桶布局的函数 ---
def generate_cone_layout():
    """根据全局配置，动态计算6个锥桶的位姿，横向距离随机"""
    rospy.loginfo("动态计算锥桶布局...")
    
    distribution_pattern = [2, 2, 1, 1]
    random.shuffle(distribution_pattern)
    
    generated_poses = []
    current_x = START_X
    
    for group_size in distribution_pattern:
        x_increment = random.uniform(MIN_GROUP_DISTANCE_X, MIN_GROUP_DISTANCE_X + 1.5)
        current_x += x_increment
        
        # 为每一组/每一个锥桶生成一个随机的横向距离
        random_y_offset = random.uniform(LATERAL_OFFSET_Y_MIN, LATERAL_OFFSET_Y_MAX)
        
        if group_size == 2:
            rospy.loginfo(f"  - 在 x={current_x:.2f} 处生成一组障碍物 (y偏移: +/- {random_y_offset:.2f})")
            generated_poses.append({'x': current_x, 'y': 4 + random_y_offset, 'z': 0, 'Yaw_degrees': 0})
            generated_poses.append({'x': current_x, 'y': 4 - random_y_offset, 'z': 0, 'Yaw_degrees': 0})
        elif group_size == 1:
            side = random.choice([-1, 1])
            y_pos = side * random_y_offset
            rospy.loginfo(f"  - 在 x={current_x:.2f} 处生成一个障碍物 (在 y={y_pos:.2f})")
            generated_poses.append({'x': current_x, 'y': 4 + y_pos, 'z': 0, 'Yaw_degrees': 0})
            
    return generated_poses

if __name__ == '__main__':
    rospy.init_node('multi_model_scene_director')

    if len(SPAWN_CONFIG_A) < NUM_MODELS_A or \
       len(SPAWN_CONFIG_B) < NUM_MODELS_B or \
       len(SPAWN_CONFIG_C) < NUM_MODELS_C or \
       len(SPAWN_CONFIG_D_QR) < NUM_MODELS_D_QR:
        rospy.logerr("错误: 固定配置中的位姿数量少于要生成的模型数量！")
    else:
        rospy.loginfo("等待Gazebo服务...")
        rospy.wait_for_service('/gazebo/delete_model')
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.loginfo("所有服务已连接。")
        
        # 1. 清理旧场景
        delete_all_spawned_models()
        rospy.sleep(1)

        # 2. 生成固定布局的模型
        process_and_spawn_models('A', NUM_MODELS_A, MODEL_LIBRARY_A, SPAWN_CONFIG_A)
        process_and_spawn_models('B', NUM_MODELS_B, MODEL_LIBRARY_B, SPAWN_CONFIG_B)
        process_and_spawn_models('C', NUM_MODELS_C, MODEL_LIBRARY_C, SPAWN_CONFIG_C)
        process_and_spawn_models('D_QR', NUM_MODELS_D_QR, MODEL_LIBRARY_D_QR, SPAWN_CONFIG_D_QR)
        
        # 3. 程序化生成并放置锥桶
        cone_poses = generate_cone_layout()
        process_and_spawn_models('E_CONE', NUM_MODELS_E_CONE, MODEL_LIBRARY_E_CONE, cone_poses)

        rospy.loginfo("所有模型生成完毕！场景构建完成。")
