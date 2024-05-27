# with open("new_train.json", "r", encoding="utf-8") as f:
#     data = f.read()

# import json

# data = json.loads(data)
# # 把列表中数据打散拆分成两份，一份占比8，一份占比2
# from sklearn.model_selection import train_test_split

# train, test = train_test_split(data, test_size=0.2)

# with open("classify_train.json", "w", encoding="utf-8") as f:
#     json.dump(train, f, ensure_ascii=False, indent=4)

# with open("classify_test.json", "w", encoding="utf-8") as f:
#     json.dump(test, f, ensure_ascii=False, indent=4)


import json
from sklearn.model_selection import train_test_split


def transform_keywords_dataset():
    with open('nl2sql_train_data.json', 'r', encoding='utf-8') as f:
        data = f.readlines()
    data = [json.loads(item) for item in data]
    final_result = []
    for item in data:
        obj = {
            "conversations": [
                {
                    "from": "human",
                    "value": item['question']
                },
                {
                    "from": "gpt",
                    "value": item['answer']
                }
            ]
        }
        final_result.append(obj)

    train, test = train_test_split(final_result, test_size=0.2)

    with open("nl2sql_train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=4)

    with open("nl2sql_test.json", "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    transform_keywords_dataset()