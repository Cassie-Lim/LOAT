import pickle
import argparse
import json
from glob import glob
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dn_startswith', type=str, required=True)
parser.add_argument('--json_name', type=str)
parser.add_argument('--results_path', type=str, default='../results/')

args = parser.parse_args()

if args.json_name is None:
    args.json_name = args.dn

lens = list()
all_actions = set()
results = {'tests_unseen': [{'trial_T20190908_010002_441405': [{'action': 'LookDown_15', 'forceAction': True}]}],
           'tests_seen': [{'trial_T20190909_042500_949430': [{'action': 'LookDown_15', 'forceAction': True}]}]}
seen_strs = ['seen', 'unseen']
# seen_strs = ['unseen']
# seen_strs = ['seen']
# ***************
key_set = set()
# ***********
for seen_str in seen_strs:
    pickle_globs = glob(args.results_path + "leaderboard/actseqs_tests_" +
                        seen_str + "_" + args.dn_startswith + "_" + "*")
    # pickle_globs = glob(args.results_path + "leaderboard/actseqs_val_" + args.dn_startswith + "_" + "*")
    print(f"the file matched are {pickle_globs}")

    pickles = []
    for g in pickle_globs:
        pickles += pickle.load(open(g, 'rb'))

    total_logs = []
    ep_num = list()
    for i, t in enumerate(pickles):
        key = list(t.keys())[0]
        # ***************
        if key not in key_set:
        # ***************
            key_set.add(key)
            actions = t[key]
            trial = key[1]
            total_logs.append({trial: actions})
            ep_num.append(key[0])

    for i, (t, ep_n) in enumerate(zip(total_logs, ep_num)):
        key = list(t.keys())[0]
        actions = t[key]
        new_actions = []
        for indx, action in enumerate(actions):
            if action['action'] == 'LookDown_0' or action['action'] == 'LookUp_0':
                pass
            else:
                all_actions.add(action['action'])
                new_actions.append(action)

            # if indx > 950:
            #     break
        assert len(new_actions) < 1000
        lens.append(indx)
        total_logs[i] = {key: new_actions}

    print(max(lens))

    print(len(total_logs))
    if seen_str == 'seen':
        # assert len(total_logs) == 1533
        if len(total_logs) != 1533:
            print(f"the spiltt is {seen_str}, the length is {len(total_logs)}, not the true length of 1533")
            # 将ep_num排序后保存
            ep_num.sort()
            # 保存ep_num文件
            with open(f'leaderboard_jsons/{seen_str}_ep_num.json', 'w') as f:
                json.dump(ep_num, f)
            # 找出不再ep_num里的轨迹
            no_ep = [i for i in range(0,1533) if i not in ep_num]
            print(no_ep)
        results['tests_seen'] = total_logs
    else:
        # assert len(total_logs) == 1529, f"the length is {len(total_logs)}, not the true length of 1529"
        if len(total_logs) != 1529:
            print(f"the spiltt is {seen_str}, the length is {len(total_logs)}, not the true length of 1529")
            # 将ep_num排序后保存
            ep_num.sort()
            # 保存ep_num文件
            with open(f'leaderboard_jsons/{seen_str}_ep_num.json', 'w') as f:
                json.dump(ep_num, f)
            # 找出不再ep_num里的轨迹
            no_ep = [i for i in range(0,1529) if i not in ep_num]
            print(no_ep)
        results['tests_unseen'] = total_logs

print(len(results['tests_seen']))
print(len(results['tests_unseen']))
print(len(all_actions))
print(all_actions)

if not os.path.exists('leaderboard_jsons'):
    os.makedirs('leaderboard_jsons')

save_path = 'leaderboard_jsons/tests_actseqs_' + args.json_name + '.json'
with open(save_path, 'w') as r:
    json.dump(results, r, indent=4, sort_keys=True)
