import torch
from tqdm import tqdm
from datasets import load_dataset
import argparse
import time
import wandb
import argparse
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import openai
import tenacity
from tenacity import retry, wait_random_exponential
import json
from collections import OrderedDict
import re
import os
from GPTtemplate import GPT_TEMPLATE,GPT_FEW_SHOT_TEMPLATE,GPT_NEGATION_TEMPLATE

with open('key.txt') as f:
    lines = f.readlines()
openai.api_key = lines[0].split('\n')[0]


@retry(wait=wait_random_exponential(min=40, max=60), stop=tenacity.stop_after_attempt(10))
def Corrector(text, MODEL):
    # Question='Can you change the text style of Answer in the style writing of Response? The logical flow and final answer of Answer should be preserved.'
    if 'gpt' in MODEL.lower():
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": text},
            ],

            stop=["\"\"\""]
        )
        message = response["choices"][0]["message"]["content"]

    elif MODEL == "text-davinci-003":
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt="Question: " + text + "Let's think step-by-step.",
            temperature=0.5,
            max_tokens=512,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.6,
            stop=["\"\"\""]
        )
        message = response["choices"][0]["text"]

    return message


def extract_question(question):
    match = re.search(r'Question:(.*?)(?=\n Let\'s think step-by-step)', question)

    # Check if there is a match
    if match:
        extracted_string = match.group(1).strip()

    else:
        print("No match found.")
    return extracted_string


def get_argument():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--mode', default='gold')
    parser.add_argument('--sample_path',
                        default='/userhomes/minsu/Holism/src/holism_data/holism.jsonl',
                        type=str)
    parser.add_argument('--output_path',
                        default='/userhomes/minsu/Holism/src/holism_data/revision/holism_revision_task.jsonl', type=str)
    parser.add_argument('--method', default="negation", type=str)
    parser.add_argument('--model', default="gpt-3.5-turbo-1106", type=str)
    args = parser.parse_args()

    return args


def main(args):
    MODEL = args.model
    data = []
    with open(args.sample_path, 'r') as f:
        for line in f:
            ex = json.loads(line)
            data.append(ex)

    f.close()

    # os.makedirs(args.output_path, exist_ok=True)
    with open(args.output_path, "a", encoding="utf-8") as f:
        for i, entry in tqdm(enumerate(data)):
            if 'negation' in args.method:
                text_template = GPT_NEGATION_TEMPLATE.format(source=entry["Explanation2"])
                print("text_template",text_template)
                output_txt = Corrector(text_template, MODEL)
                print("##output_text",output_txt)
                generated_sample = OrderedDict()
                generated_sample['Scientific fact'] = entry["Scientific fact"]
                generated_sample['Counter Observation'] = entry["Counter Observation"]
                generated_sample['Explanation1'] = entry['Explanation1']
                generated_sample['Explanation2']=output_txt
                json.dump(generated_sample, f)  #
                f.write("\n")

            else:
                text_template = GPT_FEW_SHOT_TEMPLATE.format(source=entry["sentence"])
                print("text_tempalte",text_template)
                output_txt = Corrector(text_template, MODEL)
                print("input",text_template)
                print("##output_text",output_txt)
                generated_sample = OrderedDict()
                generated_sample['Scientific fact'] = entry["sentence"]
                generated_sample['API'] = output_txt
                generated_sample['uid'] = entry['uid']
                json.dump(generated_sample, f)  #
                f.write("\n")


if __name__ == '__main__':
    main(get_argument())
    # wandb.finish()
