FROM_IDX=$1
TO_IDX=$2
SPLIT=$3
NAME=$4
SEM_POLICTY_TYPE=$5
MLM_OPTIONS=$6
LANG_GRANULARITY=${7}
GPU_ID=$8


# example usage:
# add_bycyw/scripts/inference_id.sh 0 300 valid_unseen prompter512_resize mlm "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" gt 2

# add_bycyw/scripts/inference_id.sh 0 300 valid_unseen prompter1024_resize mlm "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" gt 2

# CUDA_VISIBLE_DEVICES=1 add_bycyw/scripts/inference_id.sh 0 300 valid_unseen test_blocking mlm "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" gt 0


python add_bycyw/code/main.py \
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
--run_idx_file add_bycyw/data/selected_data/selected_data_valid_unseen.json \
--result_file add_bycyw/results_exp/ \
