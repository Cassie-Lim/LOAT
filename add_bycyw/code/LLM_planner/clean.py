"""
将chatgpt的输出转换为想要的格式
"""

if __name__ == "__main__":
    while(True):
        print("请输入chatgpt的输出：")
        chatgpt_output = ""
        while(True):
            input_now = input()
            if input_now == "end":
                break
            chatgpt_output += input_now + "\n"
        print(f"the response of chatgpt is {chatgpt_output}")

        # 处理输入
        response_list = chatgpt_output.split("\n")
        response_list = [i.split(":")[0].strip() for i in response_list]
        respose = ", ".join(response_list)
        # respose[-1]="."
        print(f"the response of chatgpt is: {respose}")

        