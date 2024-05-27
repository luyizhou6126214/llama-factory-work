import json
import time
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModel
from transformers.generation.utils import GenerationConfig

GPU_DEVICE_INDEX_4 = 4

class ChatToBaichuan7B(object):

    def __init__(self, model_path):
        torch.cuda.set_device(GPU_DEVICE_INDEX_4)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                          device_map="cuda:4", 
                                                          torch_dtype=torch.bfloat16, 
                                                          trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)

    def chat(self, message):
        # print(message)
        s_time = time.time()
        messages = []
        messages.append({"role": "user", "content": message})
        response = self.model.chat(self.tokenizer, messages)
        print(response)
        print(f'cost time:{time.time() - s_time}')
        print('\n')
        return response
    
    def run(self, test_json_path, save_path):
        import pandas as pd
        with open(test_json_path, 'r', encoding='utf-8') as f:
            data = f.read()
        data = json.loads(data)
        data = [{"question": item['conversations'][0]['value'], "answer": item['conversations'][1]['value']} for item in data]
        df_train = pd.DataFrame.from_records(data)

        # 遍历df_train，调用chat方法处理question_prompt字段，将返回结果添加到df新的一列中
        df_train['test_answer'] = df_train['question'].apply(self.chat)
        # 遍历df_train，如果每一行的0-query和query相等，返回‘’，不想等返回False,将返回结果添加到df新的一列0-query-result中
        df_train['test_result'] = df_train.apply(lambda x: True if x['test_answer'] == x['answer'] else False, axis=1)
        # 结果保存为csv文件
        df_train.to_csv(save_path, index=False)
        print('done')


if __name__ == "__main__":
    # # classify模型
    # model_path = '/workspace/luyizhou/LLaMA-Factory/models/Baichuan2-7B-chat_lora_sft-classify'
    # test_json_path = 'data/classify_test.json'
    # save_path = 'atest/test_baichuan2-7b-classify.csv'

    # # keywords模型
    # model_path = '/workspace/luyizhou/LLaMA-Factory/models/Baichuan2-7B-chat_lora_sft-keywords'
    # test_json_path = 'data/keywords_test.json'
    # save_path = 'atest/test_baichuan2-7b-keywords.csv'

    # nl2sql模型
    model_path = '/workspace/ai/model/baichuan/Baichuan2-7B-chat_lora_sft-nl2sql'
    test_json_path = 'data/nl2sql_test.json'
    save_path = 'atest/test_baichuan2-7b-nl2sql.csv'

    # # nl2sql模型
    # model_path = '/workspace/luyizhou/LLaMA-Factory/models/Baichuan2-13B-chat_lora_sft-nl2sql'
    # test_json_path = 'data/nl2sql_test.json'
    # save_path = 'atest/test_baichuan2-13b-nl2sql.csv'

    model = ChatToBaichuan7B(model_path)
    model.run(test_json_path, save_path)


