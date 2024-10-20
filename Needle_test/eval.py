import yaml
import os
import json
import re

import time
import requests

api_key = "" # Enter your openai api key here

def pred_openai(model_name, msg):
    tries = 0
    while tries < 5:
        tries += 1
        try:
            headers = {
                'Authorization': f"Bearer {api_key}"
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", json = {
                "model": model_name,
                "messages": msg,
                "temperature": 0.
            }, headers=headers, timeout=120)
            if resp.status_code != 200:
                raise Exception(resp.text)
            resp = resp.json()
            break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if "maximum context length" in str(e):
                raise e
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return
    
    return resp["choices"][0]["message"]["content"]


USER_TEMPLATE = '''[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. {criteria}[Ground truth]\n{reference}\nBegin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[Question]\n{input}\n\n[The Start of Assistant\'s Answer]\n{prediction}\n[The End of Assistant\'s Answer]'''
SYSTEM_TEMPLATE = 'You are a helpful assistant.'
CRITERIA = {
    "accuracy": """
    Score 1: The answer is completely unrelated to the reference.
    Score 3: The answer has minor relevance but does not align with the reference.
    Score 5: The answer has moderate relevance but contains inaccuracies.
    Score 7: The answer almost aligns with the reference but has minor omissions.
    Score 10: The answer is completely accurate and includes all relevant information from the reference.
    Only respond with a numberical score
    """
}

def get_criteria():
    cri = 'For this evaluation, you should primarily consider the following criteria:\n'
    for key, value in CRITERIA.items():
        cri += f'{key}: {value}\n'

    return cri

def get_user_template(input, prediction, reference, criteria):
    return USER_TEMPLATE.format(
        input=input,
        prediction=prediction,
        reference=reference,
        criteria=criteria
    )

if __name__ == '__main__':
    with open('config-eval.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    pred_dir = config['pred_dir']
    save_dir = config['save_dir']
    model_name = config['model']['model_name']
    model_provider = config['model']['model_provider']
    eval_num_needles = config['num_needles']
    criteria = get_criteria()
    # reference = " ".join(config['prompt']['needles'])
    input = config['prompt']['retrieval_question']
    # reference = []
    # for needle in config['prompt']['needles']:
    #     match=re.search(r'(.+) is one of', needle)
    #     if match:
    #         reference.append(match.group(1))

    # print(reference)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    result_dict = {}

    reference = []
    with open(f"./needles/{eval_num_needles}keys_len_4.json", 'r') as f:
        needles = json.load(f)["needles"]
    for needle in needles:
        match=re.search(r' (.+) is one of', needle)
        if match:
            reference.append(match.group(1))

    for filename in os.listdir(pred_dir):
        match = re.search(r'len_(\d+)', filename)
        if match:
            context_length = int(match.group(1))
        else:
            continue
        match = re.search(r'(\d+)needle', filename)
        if match:
            num_needles = int(match.group(1))
        else:
            continue
        if num_needles != eval_num_needles:
            continue
        if not filename.endswith('.txt'):
            continue

        with open(f'{pred_dir}/{filename}', 'r') as f:
            data = f.read().strip()
        prediction = data
        # user_template = get_user_template(input, prediction, reference, criteria)
        # if model_provider == 'OpenAI':
        #     msg = [{
        #             "role": "system",
        #             "content": SYSTEM_TEMPLATE
        #         }, {
        #             "role": "user",
        #             "content": user_template
        #         }
        #     ]
        #     result = pred_openai(model_name, msg)
            
        # else:
        #     raise NotImplementedError(f'Not implemented model provider: {model_provider}')
        
        # pattern = r"\[\[(\d+)\]\]"
        # match = re.search(pattern, result)
        # score = int(match.group(1)) if match else None
        score = 0
        for ref in reference:
            if ref in prediction:
                score += 1
        score = score / len(reference)
        # score = 1 if reference in prediction else 0
        result_dict[filename.replace('.txt', '')] = {
            'prediction': prediction,
            'score': score
        }
    print(reference)

    with open(f'{save_dir}/{model_provider}_{model_name}_{eval_num_needles}needles_eval.json', 'w') as f:
        json.dump(result_dict, f, indent=4)

