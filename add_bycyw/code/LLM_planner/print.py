import json

if __name__ == "__main__":
    file = "/raid/cyw/task_planning/prompter_cyw/prompter/add_byme/code/LLM_planner/data/prompt_valid_unseencwb_RoboGPT_valid.json"
    data = json.load(open(file, "r"))
    id_indata = 0
    while(True):
        input_str = input("")
        if input_str == "id":
            id_traj = int(input("id: "))
            id_list = [item["id"] for item in data]
            id_indata = id_list.index(id_traj)
            prompter = data[id_indata]["prompter"]
        elif input_str == "next":
            id_indata += 1
            prompter = data[id_indata]["prompter"]
            id_traj = data[id_indata]["id"]

        print(f"the id is {id_traj}\n{prompter}")

