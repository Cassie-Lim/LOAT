import skimage.morphology
import numpy as np
import copy


def _which_direction(start_o):
    yx = [0, 0]  # move forward to which direction
    remainder = start_o % (2*180)  # Becomes positive number
    if remainder <= 1e-3 or abs(remainder - (2*180)) <= 1:  # no rotation
        yx = [0, 1]
    elif abs(remainder - 180/2) <= 1:  # rotated once to the left #1.507
        yx = [1, 0]
    elif abs(remainder + 180/2 - 2*180) <= 1:  # rotated once to the right  #4.712
        yx = [-1, 0]
    elif abs(remainder - 180) <= 1:
        yx = [0, -1]
    else:
        raise Exception(
            "start_o falls into nowhere!, start_o is " + str(start_o))
    return yx


def _check_five_pixels_ahead_map_pred_for_moving(args, grid, start, direction_deg):
    dir_vec = np.array(_which_direction(direction_deg))
    grid_copy = copy.deepcopy(grid)  # for debugging
    range_val = args.side_step_step_size

    if dir_vec[0] == 0:
        cross = [1, 0]
    else:
        cross = [0, 1]

    traversible = True
    width = args.sidestep_width if hasattr(args, 'sidestep_width') else 4
    for i in range(1, range_val):
        dir_vec_new = dir_vec * i
        for j in range(-width, width + 1):
            y = j*cross[0] + start[0] + dir_vec_new[0]
            x = j*cross[1] + start[1] + dir_vec_new[1]
            # *****************
            # 运行中报错,添加一些更改代码
            x = max(0,min(x,239))
            y = max(0,min(y,239))
            # ***************
            grid_copy[y, x] = 0.5
            if grid[y, x] == 0:
                traversible = False

    return traversible


# goal_loc is (i,j)
def _add_cross_dilation_one_center(goal,  goal_loc, magnitude, additional_thickness):
    i = goal_loc[0]
    j = goal_loc[1]
    wheres_i = [a for a in range(
        i-magnitude, i+magnitude+1)] + [i] * (2*magnitude+1)
    wheres_j = [j] * (2*magnitude+1) + \
        [a for a in range(j-magnitude, j+magnitude+1)]
    for th in range(additional_thickness):
        i_new = i+th
        j_new = j+th
        wheres_i += [a for a in range(i-magnitude,
                                      i+magnitude+1)] + [i_new] * (2*magnitude+1)
        wheres_j += [j_new] * (2*magnitude+1) + \
            [a for a in range(j-magnitude, j+magnitude+1)]
        i_new = i-th
        j_new = j-th
        wheres_i += [a for a in range(i-magnitude,
                                      i+magnitude+1)] + [i_new] * (2*magnitude+1)
        wheres_j += [j_new] * (2*magnitude+1) + \
            [a for a in range(j-magnitude, j+magnitude+1)]

    wheres = (np.array(wheres_i), np.array(wheres_j))
    # remove those that exceed 0, 242
    wheres_i_new = []
    wheres_j_new = []
    for i, j in zip(wheres[0], wheres[1]):
        if i >= 0 and i <= 241 and j >= 0 and j <= 241:
            wheres_i_new.append(i)
            wheres_j_new.append(j)
    wheres = (np.array(wheres_i_new), np.array(wheres_j_new))
    goal[wheres] = 1
    return goal


def _add_cross_dilation(goal, magnitude, additional_thickness):
    goal_locs = np.where(goal != 0)
    for a in zip(goal_locs[0], goal_locs[1]):
        g = (a[0], a[1])
        goal = _add_cross_dilation_one_center(
            goal,  g, magnitude, additional_thickness)
    return goal


def _where_connected_to_curr_pose(start, traversible, seed, visited):
    non_traversible = 1 - traversible*1
    if traversible[start[0]+1, start[1]+1] == 0:
        count = 0
        while traversible[start[0]+1, start[1]+1] == 0 and count < 100:
            np.random.seed(seed + count)
            start_idx = np.random.choice(len(np.where(visited == 1)[0]))
            start = (np.where(visited == 1)[0][start_idx], np.where(
                visited == 1)[1][start_idx])
            count += 1

    connected_regions = skimage.morphology.label(traversible, connectivity=2)
    where_start_connected = np.where(
        connected_regions == connected_regions[start[0]+1, start[1]+1])
    wc_wrong = len(where_start_connected[0]) < len(np.where(visited)[0]) or np.sum(
        traversible[where_start_connected]) < np.sum(non_traversible[where_start_connected])

    if wc_wrong:  # error in connected region
        count = 0
        while wc_wrong and count < min(len(np.where(visited == 1)[0]), 100):
            start = (np.where(visited == 1)[
                     0][count], np.where(visited == 1)[1][count])
            where_start_connected = np.where(
                connected_regions == connected_regions[start[0]+1, start[1]+1])
            wc_wrong = len(where_start_connected[0]) < len(np.where(visited)[0]) or np.sum(
                traversible[where_start_connected]) < np.sum(non_traversible[where_start_connected])
            count += 1
    return where_start_connected


def _planner_broken(fmm_dist, goal, traversible, start,  seed, visited):
    where_connected = _where_connected_to_curr_pose(
        start, traversible,  seed, visited)
    return fmm_dist[start[0]+1, start[1]+1] == fmm_dist.max(), where_connected


def _get_closest_goal(start, goal):
    real_start = [start[0]+1, start[1]+1]
    goal_locs = np.where(goal == 1)
    dists = [(i-real_start[0])**2 + (j-real_start[1]) **
             2 for i, j in zip(goal_locs[0], goal_locs[1])]
    min_loc = np.argmin(dists)
    return (goal_locs[0][min_loc], goal_locs[1][min_loc])


def _block_goal(goal, original_goal):
    goal[np.where(original_goal == 1)] = 1
    return goal


def _get_center_goal(goal, pointer):
    connected_regions = skimage.morphology.label(goal, connectivity=2)
    unique_labels = [i for i in range(1, np.max(connected_regions)+1)]
    new_goal = np.zeros(goal.shape)
    centers = []
    for lab in unique_labels:
        wheres = np.where(connected_regions == lab)
        wheres_center = (int(np.rint(np.mean(wheres[0]))), int(
            np.rint(np.mean(wheres[1]))))
        centers.append(wheres_center)

    for i, c in enumerate(centers):
        new_goal[c[0], c[1]] = 1

    return new_goal, centers


def _get_approximate_success(prev_rgb, frame, action):
    wheres = np.where(prev_rgb != frame)
    wheres_ar = np.zeros(prev_rgb.shape)
    wheres_ar[wheres] = 1
    wheres_ar = np.sum(wheres_ar, axis=2).astype(bool)
    connected_regions = skimage.morphology.label(wheres_ar, connectivity=2)
    unique_labels = [i for i in range(1, np.max(connected_regions)+1)]
    max_area = -1
    for lab in unique_labels:
        wheres_lab = np.where(connected_regions == lab)
        max_area = max(len(wheres_lab[0]), max_area)
    if (action in ['OpenObject', 'CloseObject']) and max_area > 500:
        success = True
    elif max_area > 100:
        success = True
    else:
        success = False
    return success


def _append_to_actseq(success, actions, api_action):
    exception_actions = ["LookUp_30", "LookUp_0", "LookDown_30", "LookDown_0"]
    if not(api_action is None):
        action_received = api_action['action']
        if not(action_received in exception_actions):
            actions.append(api_action)
        else:
            if action_received == "LookUp_30":
                if success:
                    actions.append(dict(action="LookUp_15", forceAction=True))
                    actions.append(dict(action="LookUp_15", forceAction=True))
            elif action_received == "LookDown_30":
                if success:
                    actions.append(dict(action="LookDown_15", forceAction=True))
                    actions.append(dict(action="LookDown_15", forceAction=True))
            else:  # pass for lookup0, lookdown 0
                pass
    return actions
