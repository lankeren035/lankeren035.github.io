---
title: "GaussianModel"
date: 2026-04-07
tags: [3dgs]
categories: [3dgs]
comment: true
toc: true
published: true
permalink: "code/3dgs/scene.gaussian_model.py.GaussianModel"
hexo-path:
---

## 1. 输入

| 参数名 | 类型 | 解释 |
|---|---|---|
| sh_degree | - | - |
| optimizer_type | default='default' | - |

## 2. 属性

| 属性名 | 类型 | 来源函数 | 解释 |
|---|---|---|---|
| _features_dc | - | __init__ | - |
| _features_rest | - | __init__ | - |
| _opacity | - | __init__ | - |
| _rotation | - | __init__ | - |
| _scaling | - | __init__ | - |
| _xyz | - | __init__ | - |
| active_sh_degree | - | __init__ | - |
| denom | - | __init__ | - |
| max_radii2D | - | __init__ | - |
| max_sh_degree | - | __init__ | - |
| optimizer | - | __init__ | - |
| optimizer_type | - | __init__ | - |
| percent_dense | - | __init__ | - |
| spatial_lr_scale | - | __init__ | - |
| xyz_gradient_accum | - | __init__ | - |
| _exposure | - | create_from_pcd | - |
| _features_dc | - | create_from_pcd | - |
| _features_rest | - | create_from_pcd | - |
| _opacity | - | create_from_pcd | - |
| _rotation | - | create_from_pcd | - |
| _scaling | - | create_from_pcd | - |
| _xyz | - | create_from_pcd | - |
| exposure_mapping | - | create_from_pcd | - |
| max_radii2D | - | create_from_pcd | - |
| pretrained_exposures | - | create_from_pcd | - |
| spatial_lr_scale | - | create_from_pcd | - |
| _features_dc | - | densification_postfix | - |
| _features_rest | - | densification_postfix | - |
| _opacity | - | densification_postfix | - |
| _rotation | - | densification_postfix | - |
| _scaling | - | densification_postfix | - |
| _xyz | - | densification_postfix | - |
| denom | - | densification_postfix | - |
| max_radii2D | - | densification_postfix | - |
| tmp_radii | - | densification_postfix | - |
| xyz_gradient_accum | - | densification_postfix | - |
| tmp_radii | - | densify_and_prune | - |
| _features_dc | - | load_ply | - |
| _features_rest | - | load_ply | - |
| _opacity | - | load_ply | - |
| _rotation | - | load_ply | - |
| _scaling | - | load_ply | - |
| _xyz | - | load_ply | - |
| active_sh_degree | - | load_ply | - |
| pretrained_exposures | - | load_ply | - |
| active_sh_degree | - | oneupSHdegree | - |
| _features_dc | - | prune_points | - |
| _features_rest | - | prune_points | - |
| _opacity | - | prune_points | - |
| _rotation | - | prune_points | - |
| _scaling | - | prune_points | - |
| _xyz | - | prune_points | - |
| denom | - | prune_points | - |
| max_radii2D | - | prune_points | - |
| tmp_radii | - | prune_points | - |
| xyz_gradient_accum | - | prune_points | - |
| _opacity | - | reset_opacity | - |
| denom | - | restore | - |
| xyz_gradient_accum | - | restore | - |
| covariance_activation | - | setup_functions | - |
| inverse_opacity_activation | - | setup_functions | - |
| opacity_activation | - | setup_functions | - |
| rotation_activation | - | setup_functions | - |
| scaling_activation | - | setup_functions | - |
| scaling_inverse_activation | - | setup_functions | - |
| denom | - | training_setup | - |
| exposure_optimizer | - | training_setup | - |
| exposure_scheduler_args | - | training_setup | - |
| optimizer | - | training_setup | - |
| percent_dense | - | training_setup | - |
| xyz_gradient_accum | - | training_setup | - |
| xyz_scheduler_args | - | training_setup | - |

## 3. 方法

| 方法名 | 解释 |
|---|---|
| [[3dgs_setup_functions\|setup_functions]] | - |
| [[source/_posts/obsidian/3dgs_code_learn/functions/scene/gaussian_model.py/GaussianModel/3dgs___init__\|__init__]] | - |
| [[3dgs_capture\|capture]] | - |
| [[3dgs_restore\|restore]] | - |
| [[3dgs_get_scaling\|get_scaling]] | - |
| [[3dgs_get_rotation\|get_rotation]] | - |
| [[3dgs_get_xyz\|get_xyz]] | - |
| [[3dgs_get_features\|get_features]] | - |
| [[3dgs_get_features_dc\|get_features_dc]] | - |
| [[3dgs_get_features_rest\|get_features_rest]] | - |
| [[3dgs_get_opacity\|get_opacity]] | - |
| [[3dgs_get_exposure\|get_exposure]] | - |
| [[3dgs_get_exposure_from_name\|get_exposure_from_name]] | - |
| [[3dgs_get_covariance\|get_covariance]] | - |
| [[3dgs_oneupSHdegree\|oneupSHdegree]] | - |
| [[3dgs_create_from_pcd\|create_from_pcd]] | - |
| [[3dgs_training_setup\|training_setup]] | - |
| [[3dgs_update_learning_rate\|update_learning_rate]] | Learning rate scheduling per step |
| [[3dgs_construct_list_of_attributes\|construct_list_of_attributes]] | - |
| [[3dgs_save_ply\|save_ply]] | - |
| [[3dgs_reset_opacity\|reset_opacity]] | - |
| [[3dgs_load_ply\|load_ply]] | - |
| [[3dgs_replace_tensor_to_optimizer\|replace_tensor_to_optimizer]] | - |
| [[3dgs__prune_optimizer\|_prune_optimizer]] | - |
| [[3dgs_prune_points\|prune_points]] | - |
| [[3dgs_cat_tensors_to_optimizer\|cat_tensors_to_optimizer]] | - |
| [[3dgs_densification_postfix\|densification_postfix]] | - |
| [[3dgs_densify_and_split\|densify_and_split]] | - |
| [[3dgs_densify_and_clone\|densify_and_clone]] | - |
| [[3dgs_densify_and_prune\|densify_and_prune]] | - |
| [[3dgs_add_densification_stats\|add_densification_stats]] | - |
