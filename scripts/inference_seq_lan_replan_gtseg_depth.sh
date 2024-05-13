FROM_IDX=$1
TO_IDX=$2
SPLIT=$3
NAME=$4
MLM_OPTIONS=$5
SEQ_OPTIONS=$6
LANG_GRANULARITY=${7}
GPU_ID=${8}
X_DISPLAY=${9}
ATTN_MODE=${10}


# example usage:
# CUDA_VISIBLE_DEVICES=1 improve_search_v5/scripts/inference_seq_lan_replan.sh 0 300 tests_unseen seq "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" "lan_locs" high_low 0 21.0

python main.py \
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
--ignore_sliced \
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
--result_file results/ \
--x_display ${X_DISPLAY} \
--drop_interaction_fail_loc \
--use_replan \
--max_next_goal_request 8 \
--record_replan \
--attn_mode ${ATTN_MODE} \
# --save_pictures 