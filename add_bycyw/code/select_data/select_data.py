'''
从valid_unseen中选取一批数据，供测试其效果
选取标准：每条轨迹选取第0个标注，之后用ground truth language测试
'''
import argparse
from tqdm import tqdm
import json
import os

SPLIT_PATH = "alfred_data_small/splits/oct21.json"

def select_data(split,save_path):
    split_data = json.load(open(SPLIT_PATH,'r'))
    split_data = split_data[split]
    idxs = []
    for i,item in tqdm(enumerate(split_data)):
        if item['repeat_idx']==0:
            idxs.append(i)
    # 保存文件
    file_name = "selected_data_"+split+".json"
    json.dump(idxs,open(save_path+file_name,'w'))
    print(f"save file in {save_path+file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',type=str,choices=['valid_unseen','valid_seen','tests_unseen','tests_seen'])
    parser.add_argument('--save_path',type=str,default=None)

    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = "add_byme/data/selected_data/"
    
    if not os.path.exists(args.save_path):
        os.makedirs(os.save_path)
    
    select_data(args.split,args.save_path)




    
