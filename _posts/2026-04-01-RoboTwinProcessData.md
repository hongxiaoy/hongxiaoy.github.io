---
layout: post
title: Data Processing for PI Model in RoboTwin
date: 2026-04-01
description: A detailed guide on converting raw RoboTwin data to HDF5 format for PI model training, including state-only to state-action transformation, image encoding, and instruction processing.
tags: RoboTwin Dataset-Processing
categories: Dataset-Processing
toc:
  beginning: true
---

👋 Welcome to this blog about data processing for PI series models in RoboTwin! This post will detailedly explain how to convert raw RoboTwin format data to HDF5 format to better support PI model training and inference.

💻 The code is located in the `RoboTwin` folder, specifically in `policy/pi05/scripts/process_data.py`.

📋 The script consists of several core parts:

1. 📥 Loading the original RoboTwin data
2. 🖼️ Encoding images to bytes for HDF5 storage
3. 🔄 Creating transitions for robot demonstrations

The complete code will be provided at the end of this post for your reference.

# 📥 Loading the original RoboTwin data

The original RoboTwin data is stored in HDF5 format, and the script loads it using the `h5py` library.

This function reads joint actions, gripper actions, and images from the HDF5 file.

⚠️ Note: The images are stored in bytes datatype.

```python
def load_hdf5(dataset_path):
    # Check if the dataset directory exists
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    # ======================================
    #    Main logic to load the data
    # ======================================
    with h5py.File(dataset_path, "r") as root:
        # read all joint and gripper actions
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        # read all images
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict
```


# 🖼️ Encoding images to bytes

## 📷 Why Image Encoding Matters

**Important Note:** Storing raw images directly in HDF5 files can be inefficient in terms of both storage space and I/O performance. By encoding images to bytes, we achieve several key benefits:

1. **Reduced Storage Size:** JPEG compression significantly reduces the size of image data
2. **Consistent Data Format:** All images are converted to a uniform byte format
3. **Efficient I/O:** Byte arrays are faster to read and write compared to raw image data
4. **Fixed-Length Storage:** Padding ensures all images have the same length, which is required for HDF5 datasets

When storing images in HDF5 format for the PI model, we need to encode them to bytes first.

This process includes compressing images to JPEG format, converting them to byte arrays, and padding them appropriately to ensure all images have the same length.

```python
def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0

    # encode images to bytes format
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    
    # padding the images to the same length
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    
    return encode_data, max_len
```

# 🔄 Creating transitions for robot demonstrations

Finally, we need to create transition data for robot demonstrations.

## 🔍 Key Transformation: From State-Only to State-Action Pairs

**Important Note:** The original RoboTwin data is **state-only** - it only contains the current frame's joint positions, gripper states, and images. Here, "state" refers to the robot's **proprioceptive data** (joint angles and gripper states), which represents the robot's internal body configuration.

### How the Transformation Works

1. **Current State (t):** For each frame `j`, we capture the current proprioceptive state (joint angles + gripper states) as `state`
2. **Next State (t+1):** We use the `state` from frame `j+1` as the **action** for frame `j`
3. **State-Action Pair:** This creates a transition `(state_t, action_t)` where `action_t = state_{t+1}`

This approach effectively converts the state-only data into the state-action pairs required by PI models for training.

### Why This Matters

This transformation is crucial because PI models learn from state-action transitions to predict the next state given the current state and action. By using the next frame's state as the action, we're teaching the model to predict how the robot's body configuration evolves over time.

This process effectively constructs a complete state transition sequence that PI models can learn from.

## 📝 Processing Instructions

**Important Note:** Instructions are a critical component of the RoboTwin dataset. They provide the task context that the robot is trying to accomplish, which is essential for PI models to understand the purpose of the demonstrated actions.

The script extracts instructions from JSON files and saves them alongside the processed data, ensuring that each episode's context is preserved for model training.

```python
def data_transform(path, episode_num, save_path):
    begin = 0
    floders = os.listdir(path)
    # assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # ======================================
    #    Main logic to transform the data
    # ======================================
    for i in range(episode_num):
        
        # save seen instructions in json file
        desc_type = "seen"
        instruction_data_path = os.path.join(path, "instructions", f"episode{i}.json")
        with open(instruction_data_path, "r") as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = instruction_dict[desc_type]
        save_instructions_json = {"instructions": instructions}

        os.makedirs(os.path.join(save_path, f"episode_{i}"), exist_ok=True)

        with open(
                os.path.join(os.path.join(save_path, f"episode_{i}"), "instructions.json"),
                "w",
        ) as f:
            json.dump(save_instructions_json, f, indent=2)

        # load the joint actions, gripper actions, and images
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = (load_hdf5(
            os.path.join(path, "data", f"episode{i}.hdf5")))

        # maintain lists of the desired data field for each episode
        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        last_state = None
        for j in range(0, left_gripper_all.shape[0]):

            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )

            state = np.array(left_arm.tolist() + [left_gripper] + right_arm.tolist() + [right_gripper])  # joints angle

            state = state.astype(np.float32)

            # ============================================================================
            # save all joint and gripper actions and images except the last one frame
            # ============================================================================
            if j != left_gripper_all.shape[0] - 1:
                qpos.append(state)

                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_resized = cv2.resize(camera_high, (640, 480))
                cam_high.append(camera_high_resized)

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            # ==============================================================================================
            # save the next(current) joint and gripper states as the action of the current(last) frame
            # ==============================================================================================
            # j != 0 to avoid the first frame
            if j != 0:
                action = state
                actions.append(action)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        hdf5path = os.path.join(save_path, f"episode_{i}/episode_{i}.hdf5")
        
        # save current episode of robot demonstrations in hdf5 file
        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))  # actions
            obs = f.create_group("observations")  # obaservations
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")  # images
            cam_high_enc, len_high = images_encoding(cam_high)
            cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
            cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
            image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
            image.create_dataset("cam_right_wrist", data=cam_right_wrist_enc, dtype=f"S{len_right}")
            image.create_dataset("cam_left_wrist", data=cam_left_wrist_enc, dtype=f"S{len_left}")

        begin += 1
        print(f"proccess {i} success!")

    return begin
```


# 📝 Complete Code

Here's the complete code implementation for your reference:

```python
import sys

import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import yaml, json


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def data_transform(path, episode_num, save_path):
    begin = 0
    floders = os.listdir(path)
    # assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):
        
        desc_type = "seen"
        instruction_data_path = os.path.join(path, "instructions", f"episode{i}.json")
        with open(instruction_data_path, "r") as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = instruction_dict[desc_type]
        save_instructions_json = {"instructions": instructions}

        os.makedirs(os.path.join(save_path, f"episode_{i}"), exist_ok=True)

        with open(
                os.path.join(os.path.join(save_path, f"episode_{i}"), "instructions.json"),
                "w",
        ) as f:
            json.dump(save_instructions_json, f, indent=2)

        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = (load_hdf5(
            os.path.join(path, "data", f"episode{i}.hdf5")))
        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        last_state = None
        for j in range(0, left_gripper_all.shape[0]):

            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )

            state = np.array(left_arm.tolist() + [left_gripper] + right_arm.tolist() + [right_gripper])  # joints angle

            state = state.astype(np.float32)

            if j != left_gripper_all.shape[0] - 1:
                qpos.append(state)

                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_resized = cv2.resize(camera_high, (640, 480))
                cam_high.append(camera_high_resized)

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            if j != 0:
                action = state
                actions.append(action)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        hdf5path = os.path.join(save_path, f"episode_{i}/episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")
            cam_high_enc, len_high = images_encoding(cam_high)
            cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
            cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
            image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
            image.create_dataset("cam_right_wrist", data=cam_right_wrist_enc, dtype=f"S{len_right}")
            image.create_dataset("cam_left_wrist", data=cam_left_wrist_enc, dtype=f"S{len_left}")

        begin += 1
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        default="beat_block_hammer",
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("setting", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        default=50,
        help="Number of episodes to process (e.g., 50)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    setting = args.setting
    expert_data_num = args.expert_data_num

    load_dir = os.path.join("../../data", str(task_name), str(setting))

    begin = 0
    print(f'read data from path:{os.path.join("data", load_dir)}')

    target_dir = f"processed_data/{task_name}-{setting}-{expert_data_num}"
    begin = data_transform(
        load_dir,
        expert_data_num,
        target_dir,
    )

```

## 🎉 Conclusion

And that's a wrap! We've successfully walked through the complete data processing pipeline for PI models in RoboTwin.

This script efficiently converts raw RoboTwin data into a structured HDF5 format, ready for PI model training. The key steps include loading the original data, encoding images to bytes, and creating proper state transitions for robot demonstrations.

By following this process, you can ensure that your RoboTwin data is properly formatted for PI series models, which can lead to more effective robot learning and better performance in various tasks.

Feel free to adapt this code to your specific needs and explore how it can be extended for other robot learning scenarios. Happy coding! 🤖✨

