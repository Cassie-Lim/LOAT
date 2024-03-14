# 序列搜索，使用high level language作为辅助,使用gt language, depth, segmentation
FROM_IDX=$1
TO_IDX=$2
SPLIT=$3
NAME=$4
MLM_OPTIONS=$5
SEQ_OPTIONS=$6
LANG_GRANULARITY=${7}
GPU_ID=${8}


# example usage:
# CUDA_VISIBLE_DEVICES=1 improve_search_v4/scripts/inference_seq_lan_gtdepth_gtseg_id.sh 0 300 valid_unseen seq_high_low_v4_1 "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" "lan_locs high_low" gt 0

# CUDA_VISIBLE_DEVICES=1 improve_search_v4/scripts/inference_seq_lan_gtdepth_gtseg_id.sh 0 100 valid_unseen seq_high_low_v4_1 "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" "lan_locs high_low" gt 0
# CUDA_VISIBLE_DEVICES=1 improve_search_v4/scripts/inference_seq_lan_gtdepth_gtseg_id.sh 100 200 valid_unseen seq_high_low_v4_1 "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" "lan_locs high_low" gt 0
# CUDA_VISIBLE_DEVICES=1 improve_search_v4/scripts/inference_seq_lan_gtdepth_gtseg_id.sh 200 300 valid_unseen seq_high_low_v4_1 "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" "lan_locs high_low" gt 0


python improve_search_v4/main.py \
-n1 \
--max_episode_length 1000 \
--num_local_steps 25 \
--num_processes 1 \
--eval_split ${SPLIT} \
--from_idx ${FROM_IDX} \
--to_idx ${TO_IDX} \
--max_fails 10 \
--debug_local \
--set_dn ${NAME} \
--which_gpu ${GPU_ID} \
--depth_gpu ${GPU_ID} \
--sem_seg_gpu ${GPU_ID} \
--sem_gpu_id ${GPU_ID} \
--sem_policy_type seq \
--mlm_fname mlmscore_gpt \
--mlm_options ${MLM_OPTIONS} \
--seq_options ${SEQ_OPTIONS} \
--seed 1 \
--splits alfred_data_small/splits/oct21.json \
--grid_sz 240 \
--mlm_temperature 1 \
--approx_last_action_success \
--language_granularity ${LANG_GRANULARITY} \
--centering_strategy local_adjustment \
--target_offset_interaction 0.5 \
--obstacle_selem 9 \
--run_idx_file add_byme/data/selected_data/selected_data_valid_unseen.json \
--result_file improve_search_v4/results_exp/ \
--x_display 0 \
--drop_interaction_fail_loc \