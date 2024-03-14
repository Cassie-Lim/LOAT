DISPLAY=0.0
# LANG_GRANULARITY="high_low"
LANG_GRANULARITY="high"
ATTN_MODE="cap_avg_auto"
ATTN_MODE="cap_avg"
# ATTN_MODE="cap_mul"
# SET=tests_unseen
# SET=tests_seen 
SET=valid_unseen  # 821
SET=valid_seen  # 820
SET_DN=cap_avg_0_tmp
# SET_DN=cap_avg_1
# SET_DN=cap_avg_0.5
# SET_DN=cap_avg_auto_v1_noexp # do not * explored_all
# SET_DN=cap_avg_auto_v2_tmp
IDX_ID=$1
# GAP=200
GAP=103
GAP=10
BASE=668
# ((FROM_IDX=${GAP}*(${IDX_ID}-1)))
((FROM_IDX=${BASE}+${GAP}*(${IDX_ID}-1)))
((TO_IDX=${FROM_IDX}+${GAP}))
FROM_IDX=0
TO_IDX=1
echo $SET
echo $FROM_IDX
echo $TO_IDX
CUDA_VISIBLE_DEVICES=1 improve_search_v5/scripts/inference_seq_lan_replan_lmy.sh ${FROM_IDX} ${TO_IDX} ${SET} ${SET_DN} "aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn no_slice_replay" "lan_locs" ${LANG_GRANULARITY} 0 ${DISPLAY} ${ATTN_MODE}

# 173-308 515-617 719-821