FROM_IDX=$1
TO_IDX=$2
SPLIT=$3
NAME=$4
SEM_POLICTY_TYPE=$5
MLM_OPTIONS=$6
LANG_GRANULARITY=${7}
GPU_ID=${8}


# example usage:
# trail/yolo_mask_rcnn_small/scripts/inference_replan.sh 0 510 valid_unseen testrun mlm "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" high

# 在test上跑
# conda activate prompter
# CUDA_VISIBLE_DEVICES=1 trail/yolo_mask_rcnn_small/scripts/inference_replan.sh 0 200 tests_unseen prompter512_yolo_small mlm "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" high_low 0
# CUDA_VISIBLE_DEVICES=1 trail/yolo_mask_rcnn_small/scripts/inference_replan.sh 200 400 tests_unseen prompter512_yolo_small mlm "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" high_low 0
# CUDA_VISIBLE_DEVICES=1 trail/yolo_mask_rcnn_small/scripts/inference_replan.sh 400 600 tests_unseen prompter512_yolo_small mlm "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" high_low 0
# CUDA_VISIBLE_DEVICES=1 trail/yolo_mask_rcnn_small/scripts/inference_replan.sh 600 800 tests_unseen prompter512_yolo_small mlm "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" high_low 0
# CUDA_VISIBLE_DEVICES=2 trail/yolo_mask_rcnn_small/scripts/inference_replan.sh 800 1000 tests_unseen prompter512_yolo_small mlm "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" high_low 0
# CUDA_VISIBLE_DEVICES=2 trail/yolo_mask_rcnn_small/scripts/inference_replan.sh 1000 1200 tests_unseen prompter512_yolo_small mlm "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" high_low 0
# CUDA_VISIBLE_DEVICES=2 trail/yolo_mask_rcnn_small/scripts/inference_replan.sh 1200 1400 tests_unseen prompter512_yolo_small mlm "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" high_low 0
# CUDA_VISIBLE_DEVICES=2 trail/yolo_mask_rcnn_small/scripts/inference_replan.sh 1400 1600 tests_unseen prompter512_yolo_small mlm "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" high_low 0

# 断了重跑
#  CUDA_VISIBLE_DEVICES=1 trail/yolo_mask_rcnn_small/scripts/inference_replan.sh 767 800 tests_unseen prompter512_yolo_small mlm "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" high_low 0

python trail/yolo_mask_rcnn_small/main.py \
-n1 \
--max_episode_length 1000 \
--num_local_steps 25 \
--num_processes 1 \
--eval_split ${SPLIT} \
--from_idx ${FROM_IDX} \
--to_idx ${TO_IDX} \
--max_fails 10 \
--debug_local \
--learned_depth \
--use_sem_seg \
--ignore_sliced \
--set_dn ${NAME} \
--which_gpu ${GPU_ID} \
--depth_gpu ${GPU_ID} \
--sem_seg_gpu ${GPU_ID} \
--sem_gpu_id ${GPU_ID} \
--sem_policy_type ${SEM_POLICTY_TYPE} \
--mlm_fname mlmscore_gpt \
--mlm_options ${MLM_OPTIONS} \
--seed 1 \
--splits alfred_data_small/splits/oct21.json \
--grid_sz 240 \
--mlm_temperature 1 \
--approx_last_action_success \
--language_granularity ${LANG_GRANULARITY} \
--centering_strategy local_adjustment \
--target_offset_interaction 0.5 \
--obstacle_selem 9 \
--result_file trail/yolo_mask_rcnn_small/results_exp_tests/ \
--use_replan \
--record_replan \
--max_next_goal_request 8 \
