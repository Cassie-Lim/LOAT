'''
清洗数据的文件
'''
import json
import os
from glob import glob
import progressbar

def get_unmathched_id(file):
    '''
    获取未匹配的id
    '''
    unmatched_id = []
    data = json.load(open(file, 'r'))
    for item in data:
        unmatched_id.append(item['id'])
    return unmatched_id

def drop_item(data,drop_id,key_name="id"):
    '''
    删除数据
    '''
    data_new = []
    for item in progressbar.progressbar(data):
        if item[key_name] not in drop_id:
            data_new.append(item)
    return data_new

def drop_for_split(split,data_path=None,id_path=None,save_path=None ):
    if split not in ["train","valid_seen","valid_unseen"]:
        raise Exception("split must be train,valid_seen,valid_unseen")
    else:
        id_file = os.path.join(id_path,split+"_not_matched_list.json")
        print(f"the id file is {id_file}")
        id_todrop = get_unmathched_id(id_file)
        files = glob(os.path.join(data_path,"*"+split+".json"))
        print(f"the files are{files}")
        
        for file in files:
            data = json.load(open(file, 'r'))
            file_name = file.split("/")[-1].split(".")[0]
            if "newtype" in file_name:
                data = drop_item(data,id_todrop,"base_id")
            else:
                data = drop_item(data,id_todrop,"id")
            # 保存数据
            save_file = os.path.join(save_path,file_name+"_cleaned.json")
            json.dump(data,open(save_file, 'w'),indent=4)

def clean_data(data_path=None,id_path=None,save_path=None):
    if data_path is None:
        data_path = "/raid/cyw/Prompter_results/clean_data/data_generate_bycode"
    if id_path is None:
        id_path = "/raid/cyw/Prompter_results/clean_data/compare_gt_bert"
    if save_path is None:
        save_path = "/raid/cyw/Prompter_results/clean_data/data_cleaned"
    splits = ["train","valid_seen","valid_unseen"]
    for split in splits:
        drop_for_split(split,data_path,id_path,save_path)

if __name__ == "__main__":
    clean_data()


