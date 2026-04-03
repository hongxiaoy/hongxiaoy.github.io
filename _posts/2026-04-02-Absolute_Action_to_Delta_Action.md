---
layout: post
title: Converting Absolute Action to Delta Action
date: 2026-04-02
description: A comprehensive guide on transforming absolute action into delta action for PI series algorithms, including step-by-step process and implementation details.
tags: LeRobot Dataset-Processing
categories: Dataset-Processing
toc:
  sidebar: left
---

👋 Welcome to this guide on converting absolute action to delta action! This post will walk you through the process of transforming absolute robot actions into delta actions, a crucial step in PI series algorithms.

🔄 In PI series algorithms, converting absolute actions to delta actions is essential for effective robot learning. This transformation helps the model learn relative movements rather than absolute positions, which can lead to more robust and generalizable policies.

💻 The LeRobot implementation of this conversion can be found in `src/lerobot/datasets/lerobot_dataset.py` under the LeRobot repository root directory.

## 🔍 The Conversion Process

### Step 1: Load Absolute Actions
When loading the dataset, given a specified chunk size, we load the absolute actions for both the current timestep and future timesteps. This results in an absolute action array with shape `(chunk_size, action_dim)`, where:
- `chunk_size`: Number of consecutive timesteps
- `action_dim`: Dimensionality of the action space (including both joint and gripper actions)

### Step 2: Calculate Delta Actions
Next, we load the state of the current timestep and subtract it from the absolute action array. The result is the delta action array, which represents the relative movement needed to reach each future state.

### 🔧 Key Consideration: Selective Conversion
**Important Note:** Only the joint actions are converted to delta actions, while the gripper actions remain in their absolute form. This is because:
- Joint actions represent positions that benefit from relative movement learning
- Gripper actions typically represent binary or continuous states that are more naturally handled as absolute values

### 📊 Example Calculation

#### Inputs:
- Current state (joints only): `[0.1, -0.2, 0.3, 0.0, 0.5, -0.1]`
- Absolute action array (joints only): `[[0.15, -0.25, 0.35, 0.05, 0.55, -0.05], [0.2, -0.3, 0.4, 0.1, 0.6, 0.0]]`

#### Calculation:
```
Delta action = Absolute action - Current state
```

#### Output:
- Delta action array: `[[0.05, -0.05, 0.05, 0.05, 0.05, 0.05], [0.1, -0.1, 0.1, 0.1, 0.1, 0.1]]`

### 🎯 Benefits of Delta Actions

1. **Improved Generalization**: Delta actions allow the model to learn relative movements that can be applied across different initial positions
2. **Reduced Input Space**: The range of delta values is typically smaller than absolute positions, making learning easier
3. **Enhanced Robustness**: Relative movements are less sensitive to calibration errors and environmental changes
4. **Better Transferability**: Policies learned with delta actions can more easily transfer to different robot configurations

## 🎉 Conclusion

Converting absolute actions to delta actions is a fundamental step in PI series algorithms that significantly enhances robot learning performance. By focusing on relative movements rather than absolute positions, we enable more robust and generalizable policies.

The process is straightforward yet powerful:
1. Load absolute actions for the current and future timesteps
2. Subtract the current state to obtain delta actions
3. Maintain absolute gripper actions while converting joint actions

This transformation not only improves learning efficiency but also leads to policies that can better adapt to new situations and different robot configurations.

Whether you're working with LeRobot datasets or other robot learning frameworks, implementing this conversion can help you achieve better results in your robotic manipulation tasks. Happy coding! 🤖✨
