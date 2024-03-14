small2indx = {'AlarmClock': 0, 'Apple': 1, 'AppleSliced': 2, 'BaseballBat': 3, 'BasketBall': 4, 'Book': 5,
                   'Bowl': 6, 'Box': 7, 'Bread': 8, 'BreadSliced': 9, 'ButterKnife': 10, 'CD': 11, 'Candle': 12,
                   'CellPhone': 13, 'Cloth': 14, 'CreditCard': 15, 'Cup': 16, 'DeskLamp': 17, 'DishSponge': 18,
                   'Egg': 19, 'Faucet': 20, 'FloorLamp': 21, 'Fork': 22, 'Glassbottle': 23, 'HandTowel': 24,
                   'HousePlant': 25, 'Kettle': 26, 'KeyChain': 27, 'Knife': 28, 'Ladle': 29, 'Laptop': 30,
                   'LaundryHamperLid': 31, 'Lettuce': 32, 'LettuceSliced': 33, 'LightSwitch': 34, 'Mug': 35,
                   'Newspaper': 36,
                   'Pan': 37, 'PaperTowel': 38, 'PaperTowelRoll': 39, 'Pen': 40, 'Pencil': 41, 'PepperShaker': 42,
                   'Pillow': 43, 'Plate': 44, 'Plunger': 45, 'Pot': 46, 'Potato': 47, 'PotatoSliced': 48,
                   'RemoteControl': 49, 'SaltShaker': 50, 'ScrubBrush': 51, 'ShowerDoor': 52, 'SoapBar': 53,
                   'SoapBottle': 54, 'Spatula': 55, 'Spoon': 56, 'SprayBottle': 57, 'Statue': 58, 'StoveKnob': 59,
                   'TeddyBear': 60, 'Television': 61, 'TennisRacket': 62, 'TissueBox': 63, 'ToiletPaper': 64,
                   'ToiletPaperHanger': 65, 'ToiletPaperRoll': 66, 'Tomato': 67, 'TomatoSliced': 68, 'Towel': 69,
                   'Vase': 70, 'Watch': 71, 'WateringCan': 72, 'WineBottle': 73}
large2indx = {'ArmChair': 0, 'BathtubBasin': 1, 'Bed': 2, 'Cabinet': 3, 'Cart': 4, 'CoffeeMachine': 5,
                   'CoffeeTable': 6, 'CounterTop': 7, 'Desk': 8, 'DiningTable': 9, 'Drawer': 10,
                   'Dresser': 11, 'Fridge': 12, 'GarbageCan': 13, 'Microwave': 14, 'Ottoman': 15,
                   'Shelf': 16, 'SideTable': 17, 'SinkBasin': 18, 'Sofa': 19, 'StoveBurner': 20, 'TVStand': 21,
                   'Toilet': 22}

from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import json
import os

def load_sentences(reference_sentence):
    with open(r'data/data_with_plan.json') as f:
        # 读取 JSON 数据
        data = json.load(f)
    other_sentences = data['data']
    plan = data['plan']
    visible_object = data['visible_objects']

    return reference_sentence, other_sentences, plan, visible_object

def computate_similarity(reference_sentence, visible_object_from_RGB, similarity_method='Euclidean'):
    reference_sentence, other_sentences, plan, visible_object = load_sentences(reference_sentence)

    # Load the pre-trained BERT model and tokenizer
    # model_name = r'E:\Python_Program\navigation\try_BERT\bert-base-uncased'
    model_name = r"/home/cyw/cwb/film/prompter/add_by_cwb/model/bert-base-uncased"
    model = BertModel.from_pretrained(model_name, local_files_only=True)
    tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)

    # Tokenize and encode the reference sentence and other sentences
    encoded_inputs = tokenizer([reference_sentence] + other_sentences, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']

    # Compute BERT embeddings
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state

    # Compute reference sentence embedding
    reference_embedding = embeddings[0,0,:]

    # Compute embeddings for other sentences
    other_embeddings = embeddings[1:,0,:]

    if similarity_method == 'cosine':
        # 余弦相似度
        cosine_similarities = cosine_similarity(reference_embedding.unsqueeze(0), other_embeddings)
        final_similarity = cosine_similarities[0,:120]
    else:
        # 欧几里得相似度

        dist = F.pairwise_distance(reference_embedding.unsqueeze(0), other_embeddings)
        # Output the similarity score
        final_similarity = 1 / (1 + dist)  # Convert distance to similarity score



    sorted_numbers = sorted(enumerate(final_similarity), key=lambda x: x[1], reverse=True)

    # Get the indices of the top 8 numbers
    top_indices = [index for index, _ in sorted_numbers[:9]]

    selected_data = [other_sentences[index] for index in top_indices]

    # print("Create a high-level plan for completing a household task using the allowed actions and visible objects.")
    # print(" ")
    # print("Allowed actions: OpenObject, CloseObject, PickupObject, PutObject, ToggleObjectOn, ToggleObjectOff, SliceObject, FindObject")
    # print(" ")
    text = ""
    text += "Create a high-level plan for completing a household task using the allowed actions and visible objects.\n"
    text += " \n"
    text +="Allowed actions: OpenObject, CloseObject, PickupObject, PutObject, ToggleObjectOn, ToggleObjectOff, SliceObject, FindObject\n"
    text +="\n"



    for index in top_indices:

        # print(other_sentences[index], "/", visible_object[index], "/", plan[index])
        # print("Task description: ", other_sentences[index])
        # print("Visible objects are ", visible_object[index])
        # print("Plan: ", plan[index])
        # print(" ")
        text +=f"Task description: {other_sentences[index]}\n" 
        text += f"Visible objects are {visible_object[index]}\n" 
        text +=f"Plan: {plan[index]}\n"
        text += "\n"

    text+=f"Task description: {reference_sentence}\n"

    # text +=f"Visible objects are Knife, Microwave, Fridge, SinkBasin, Faucet, {visible_object_from_RGB}\n"
    text +=f"Visible objects are {visible_object_from_RGB}\n"
    text +="Plan:\n"
    print(text)
    return selected_data,text

def get_object_from_scene(scene):
    object_poses = scene["object_poses"]
    objects = set()
    for object_pose in object_poses:
        objectName = object_pose["objectName"]
        objectName = objectName.split("_")[0]
        objects.add(objectName)
    return objects

def get_prompt(goal_file,split="valid_unseen",save_path=None):
    goal_data = json.load(open(goal_file))
    goal_file_name = goal_file.split("/")[-1].split(".")[0]
    if save_path is None:
        save_path = "./data"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    split_file = "/raid/cyw/task_planning/prompter_cyw/prompter/alfred_data_small/splits/oct21.json"
    split_data = json.load(open(split_file, "r"))[split]
    data = []
    for item in goal_data:
        goal_description = item["high_level_instructions"]
        # 根据id读取轨迹
        traj_info = split_data[item['id']]
        repeat_idx, task = traj_info["repeat_idx"],traj_info["task"]
        json_dir = '/raid/cyw/task_planning/prompter_cyw/prompter/alfred_data_all/json_2.1.0/'+ task + '/pp/ann_' + str(repeat_idx) + '.json'
        traj_data = json.load(open(json_dir))

        # 从轨迹中读取环境中有的物体
        obejcts_scene = get_object_from_scene(traj_data["scene"])
        obejcts_scene = ", ".join(obejcts_scene)

        selected_data,prompter= computate_similarity(goal_description, obejcts_scene, similarity_method='E')

        data.append({"id":item['id'],"goal_description":goal_description,"prompter":prompter})
    # 保存文件
    json.dump(data,open(save_path+"/prompt_"+split+"_"+goal_file_name+".json","w"),indent=4)






if __name__ == "__main__":
    # get_prompt("/raid/cyw/task_planning/prompter_cyw/prompter/add_byme/data/plan_data/cwb_RoboGPT_valid.json")

    # print("over!")




    # reference_sentence = "how to put a cup on the desk by the chair"
    # visible_object_from_RGB = "mug, desk, chair"

    # reference_sentence = "cut a slice of bread, warm it with the microwave, put it on the counter along with putting the knife in the cabinet"
    # visible_object_from_RGB = "knife, bread, microwave, cabinet, counter"

    # Bring a white cup from the fridge to the counter
    # reference_sentence = "Bring a white cup from the fridge to the counter"
    # visible_object_from_RGB = "cup, fridge, counter"

    # # microwave a mug from the cupboard and put it next to the toaster
    # reference_sentence = "microwave a mug from the cupboard and put it next to the toaster"
    # visible_object_from_RGB = "microwave, cupboard, toaster, mug"

    # There is a stove and no microwave, how to heat an apple
    # reference_sentence = "There is a stove and no microwave, how to heat an apple"
    # visible_object_from_RGB = "stove, apple"

    # pick up the apple in the fridge and then put it on the table
    # reference_sentence = "pick up the apple in the fridge and then put it on the table"
    # visible_object_from_RGB = "apple, fridge, table"

    # Put three heads of lettuce on the table
    # reference_sentence = "Put three heads of lettuce on the table"
    # visible_object_from_RGB = "lettuce, table"

    # Please heat up three apples and place them on the table
    # reference_sentence = "heat up three apples and place them on the table"
    # visible_object_from_RGB = "apple, table, microwave"

    # # Put a bowl with a cleaned lettuce in it on the table
    # reference_sentence = "Put a bowl with a cleaned lettuce in it on the table"
    # visible_object_from_RGB = "bowl, lettuce, table, sinkbasin"

    # clean three apples and put them on the table
    reference_sentence = "clean three apples and put them on the table"
    visible_object_from_RGB = "apple, table, sinkbasin"




    selected_data,_ = computate_similarity(reference_sentence, visible_object_from_RGB, similarity_method='E')







