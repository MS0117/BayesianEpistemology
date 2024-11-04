import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed, AutoModelForCausalLM,TextGenerationPipeline,Text2TextGenerationPipeline
from datasets import load_dataset
from args import default_parse
import time
import wandb
import argparse
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from peft import PeftModel,PeftConfig
from evaluate import load
import datasets
import openai
import tenacity
from tenacity import retry, wait_random_exponential
import json
from collections import OrderedDict
import os
import re
import random
from collections import Counter
from template.prompt_format import ABDUCTION_ZERO_SHOT, REVISION_ZERO_SHOT,ABDUCTION_LLAMA_ZERO_SHOT,REVISION_LLAMA_ZERO_SHOT,REVISION_PHI1_ZERO_SHOT,ABDUCTION_PHI1_ZERO_SHOT,REVISION_PHI2_ZERO_SHOT,ABDUCTION_PHI2_ZERO_SHOT,GENERATION_ZERO_SHOT,GENERATION_LLAMA_ZERO_SHOT,GENERATION_PHI1_ZERO_SHOT,GENERATION_PHI2_ZERO_SHOT

with open('key.txt') as f:
    lines = f.readlines()
openai.api_key = lines[0].split('\n')[0]


@retry(wait=wait_random_exponential(min=40, max=60), stop=tenacity.stop_after_attempt(10))
def Response(text, MODEL):
    if MODEL == "gpt-3.5-turbo-0613" or MODEL== 'gpt-4':
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content":  text},
            ],
            temperature=0,
        )
        message = response["choices"][0]["message"]["content"]

    elif MODEL == "text-davinci-003":
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt= text,

        )
        message = response["choices"][0]["text"]

    return message



class Evaluator(object):
    def __init__(self, args):
        self.args = args
        if 'llama' in self.args.model_name or 'gpt2' in self.args.model_name or 't5' in self.args.model_name or 'mistral' in self.args.model_name  or 'phi' in self.args.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, cache_dir=self.args.cache_dir)
            self.tokenizer.pad_token = self.tokenizer.eos_token


        self.generation_kwargs = {"do_sample": False,  "num_return_sequences": 1,"max_new_tokens":512,  "early_stopping": True}

    def prepare_dataset(self):

        test_data = []
        with open(self.args.test_set, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                test_data.append(ex)
        f.close()

        return test_data

    def format_and_shuffle_observations(self,test_data_item, template_format, test_mode):

        if 'abduction' in test_mode:
            explanations = [(test_data_item['Explanation1'], 'A'), (test_data_item['Explanation2'], 'B')]
            random.shuffle(explanations)

            formatted_input = template_format.format(
                observation=test_data_item['Counter Observation'],
                explanation1=explanations[0][0],
                explanation2=explanations[1][0]
            )

            explanation_map = {
                explanations[0][1]: 'Explanation1',
                explanations[1][1]: 'Explanation2'
            }
        elif 'revision' in test_mode:
            explanations = [(test_data_item['Scientific fact'], 'A'), (test_data_item['Explanation2'], 'B')]
            random.shuffle(explanations)

            formatted_input = template_format.format(
                observation=test_data_item['Counter Observation'],
                explanation1=explanations[0][0],
                explanation2=explanations[1][0]
            )

            explanation_map = {
                explanations[0][1]: 'Scientific fact',
                explanations[1][1]: 'Explanation2'
            }

        return formatted_input, explanation_map

    def prepare_model(self):
        if self.args.bf16:
            self.dtype = torch.bfloat16
        elif self.args.fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        try:
            model = AutoModelForCausalLM.from_pretrained(self.args.model_name, torch_dtype=self.dtype, trust_remote_code=True, cache_dir=self.args.cache_dir, low_cpu_mem_usage=True)
        except:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name, torch_dtype=self.dtype, trust_remote_code=True, cache_dir=self.args.cache_dir, low_cpu_mem_usage=True)


        return model

    def evaluate(self):
        Num_Choose_condition=0
        directory_path = f"./3_{started_at.tm_mday}/"

        # Create the directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)
        test_data=self.prepare_dataset()
        if 't5' in self.args.model_name:
            file_name=self.args.model_name[7:]
        elif 'llama' in self.args.model_name:
            file_name= self.args.model_name[11:]
        elif 'gpt2' in self.args.model_name:
            file_name= self.args.model_name
        elif 'phi' in self.args.model_name:
            file_name = self.args.model_name[10:]
        else:
            file_name = self.args.model_name

                
        with open(f"{directory_path}{file_name}_{self.args.test_mode}.jsonl", "a", encoding="utf-8") as f:

            if 'llama' in self.args.model_name or 't5' in self.args.model_name or 'gpt2' in self.args.model_name or 'mistral' in self.args.model_name  or 'phi' in self.args.model_name:

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(device)
                model = self.prepare_model()
                if 'llama' in self.args.model_name or 'gpt2' in self.args.model_name or 'mistral' in self.args.model_name  or 'phi' in self.args.model_name:
                    text_generation_pipeline = TextGenerationPipeline(
                        model=model,
                        tokenizer=self.tokenizer,
                        torch_dtype=self.dtype,
                        device=device
                    )

                else:
                    text_generation_pipeline = Text2TextGenerationPipeline(
                        model=model,
                        tokenizer=self.tokenizer,
                        torch_dtype=self.dtype,
                        device=device
                    )

                def generate_text(prompt):
                    generated_sequences = text_generation_pipeline(
                        prompt, **self.generation_kwargs

                    )
                    # print("generated",generated_sequences)
                    return generated_sequences[0]["generated_text"].replace(prompt, "")

                peripheral_list = []
                for i in tqdm(range(len(test_data)),total=len(test_data)):

                    peripheral=0
                    if self.args.test_mode!='generation':

                        if self.args.test_mode == 'abduction':
                            if 'llama' in args.model_name or 'mistral' in args.model_name:
                                input, explanation_map = self.format_and_shuffle_observations(test_data[i], ABDUCTION_LLAMA_ZERO_SHOT,self.args.test_mode)
                            elif 't5' in args.model_name:
                                input, explanation_map = self.format_and_shuffle_observations(test_data[i], ABDUCTION_ZERO_SHOT,self.args.test_mode)
                            elif 'phi-1' in args.model_name:
                                input, explanation_map = self.format_and_shuffle_observations(test_data[i], ABDUCTION_PHI1_ZERO_SHOT,self.args.test_mode)
                            elif 'phi-2' in args.model_name:
                                input, explanation_map = self.format_and_shuffle_observations(test_data[i], ABDUCTION_PHI2_ZERO_SHOT,self.args.test_mode)

                        elif self.args.test_mode == 'revision':
                            if 'llama' in args.model_name or 'mistral' in args.model_name:
                                input, explanation_map = self.format_and_shuffle_observations(test_data[i], REVISION_LLAMA_ZERO_SHOT,self.args.test_mode)
                            elif 't5' in args.model_name:
                                input, explanation_map = self.format_and_shuffle_observations(test_data[i],REVISION_ZERO_SHOT,self.args.test_mode)
                            elif 'phi-1' in args.model_name:
                                input, explanation_map = self.format_and_shuffle_observations(test_data[i],REVISION_PHI1_ZERO_SHOT,self.args.test_mode)
                            elif 'phi-2' in args.model_name:
                                input, explanation_map = self.format_and_shuffle_observations(test_data[i],REVISION_PHI2_ZERO_SHOT,self.args.test_mode)




                        response = generate_text(input)
                        print("response",response)

                        """
                        if "Let's" in corrected_response or "let's" in corrected_response:
                            sliced_sentence_including_lets = sentence.split("Let's", 1)[1] if "Let's" in corrected_response else corrected_response.split("let's", 1)[1]
                            sliced_sentence_including_lets = "Let's" + sliced_sentence_including_lets
                        else:
                            sliced_sentence_including_lets = sentence
                        corrected_response=sliced_sentence_including_lets
                        """
                        generated_answer = OrderedDict()
                        generated_answer["input"]=input
                        generated_answer["answer"] = response
                        generated_answer["explanation_map"] = explanation_map
                        generated_answer["Scientific fact"] = test_data[i]['Scientific fact']
                        generated_answer["Counter Observation"] = test_data[i]['Counter Observation']
                        generated_answer["original_explanation1"] = test_data[i]['Explanation1']
                        generated_answer["original_explanation2"] = test_data[i]['Explanation2']
                        pattern = re.compile(r'[A-B]')

                        try:
                            answer = pattern.search(generated_answer['answer']).group() #First occurence of

                            # Check for (A) or (B) in the answer
                            if "(A)" in generated_answer['answer']:
                                answer = "A"
                            elif "(B)" in generated_answer['answer']:
                                answer = "B"
                            print("match",answer)
                            # Check if the 'answer' corresponds to 'Explanation1'

                            if 'abduction' in self.args.test_mode and generated_answer['explanation_map'][answer] == "Explanation2":
                                Num_Choose_condition += 1
                                peripheral = 1

                            elif 'revision' in self.args.test_mode  and generated_answer['explanation_map'][answer] == "Explanation2":
                                Num_Choose_condition += 1
                                peripheral = 1
                            peripheral_list.append(peripheral)

                        except:
                            pass

                    elif self.args.test_mode == 'generation':
                        if 'llama' in args.model_name or 'mistral' in args.model_name:
                            input =  GENERATION_LLAMA_ZERO_SHOT.format(hypothesis=test_data[i]['Scientific fact'], observation=test_data[i]['Counter Observation'])

                        elif 't5' in args.model_name:
                            input =  GENERATION_ZERO_SHOT.format(hypothesis=test_data[i]['Scientific fact'], observation=test_data[i]['Counter Observation'])
                        elif 'phi-1' in args.model_name:
                            input =  GENERATION_PHI1_ZERO_SHOT.format(hypothesis=test_data[i]['Scientific fact'], observation=test_data[i]['Counter Observation'])
                        elif 'phi-2' in args.model_name:
                            input =  GENERATION_PHI2_ZERO_SHOT.format(hypothesis=test_data[i]['Scientific fact'], observation=test_data[i]['Counter Observation'])

                        response = generate_text(input)
                        generated_answer = OrderedDict()
                        generated_answer["input"] = input
                        generated_answer["Scientific fact"] = test_data[i]['Scientific fact']
                        generated_answer["Counter Observation"] = test_data[i]['Counter Observation']
                        generated_answer["original_explanation1"] = test_data[i]['Explanation1']
                        generated_answer["original_explanation2"] = test_data[i]['Explanation2']
                        generated_answer["answer"] = response

                    json.dump(generated_answer, f)  #
                    f.write("\n")




            else:
                peripheral_list = []
                for i in tqdm(range(len(test_data)), total=len(test_data)):
                    peripheral = 0
                    if self.args.test_mode != 'generation':
                        if self.args.test_mode == 'abduction':
                            input, explanation_map = self.format_and_shuffle_observations(test_data[i], ABDUCTION_ZERO_SHOT,self.args.test_mode)
                        elif self.args.test_mode == 'revision':
                            input, explanation_map = self.format_and_shuffle_observations(test_data[i], REVISION_ZERO_SHOT,self.args.test_mode)

                        response = Response(input,self.args.model_name)
                        print("response", response)

                        generated_answer = OrderedDict()
                        generated_answer["input"]=input
                        generated_answer["answer"] = response
                        generated_answer["explanation_map"] = explanation_map
                        generated_answer["Scientific fact"] = test_data[i]['Scientific fact']
                        generated_answer["Counter Observation"] = test_data[i]['Counter Observation']
                        generated_answer["original_explanation1"] = test_data[i]['Explanation1']
                        generated_answer["original_explanation2"] = test_data[i]['Explanation2']
                        pattern = re.compile(r'[A-Z]')
                        try:
                            answer = pattern.search(generated_answer['answer']).group()

                            # Check for (A) or (B) in the answer
                            if 'phi-1' in self.args.model_name:
                                match = re.search(r"\(A\)|\(B\)", generated_answer['answer'])

                                if match:
                                    if match.group() == "(A)":
                                        answer = "A"
                                    elif match.group() == "(B)":
                                        answer = "B"
                                else:
                                    answer = None

                            else:
                                if "(A)" in generated_answer['answer']:
                                    answer = "A"
                                elif "(B)" in generated_answer['answer']:
                                    answer = "B"

                            # Check if the 'answer' corresponds to 'Explanation1'

                            if 'abduction' in self.args.test_mode and generated_answer['explanation_map'][answer] == "Explanation2":
                                Num_Choose_condition += 1
                                peripheral = 1

                            elif 'revision' in self.args.test_mode  and generated_answer['explanation_map'][answer] == "Explanation2":
                                Num_Choose_condition += 1
                                peripheral = 1

                            peripheral_list.append(peripheral)

                        except:
                            pass

                    elif self.args.test_mode == 'generation':

                        input = GENERATION_ZERO_SHOT.format(hypothesis=test_data[i]['Scientific fact'],
                                                                observation=test_data[i]['Counter Observation'])

                        response = Response(input,self.args.model_name)
                        print("response", response)
                        generated_answer = OrderedDict()
                        generated_answer["input"] = input
                        generated_answer["Scientific fact"] = test_data[i]['Scientific fact']
                        generated_answer["Counter Observation"] = test_data[i]['Counter Observation']
                        generated_answer["original_explanation1"] = test_data[i]['Explanation1']
                        generated_answer["original_explanation2"] = test_data[i]['Explanation2']
                        generated_answer["answer"] = response

                    json.dump(generated_answer, f)  #
                    f.write("\n")

            json.dump(peripheral_list, f)

        f.close()

        if self.args.test_mode!='generation':
            Choose_condition = Num_Choose_condition/len(test_data)

            wandb.log({'Choose_condition': Choose_condition})
            print(Choose_condition)
            print(peripheral_list)


if __name__ == '__main__':

    started_at = time.gmtime()
    parser = argparse.ArgumentParser(
        ""
    )
    args = default_parse(parser)
    seed = args.seed
    deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    wandb.init(project=args.project,
               entity=args.entity,
               name=f"{args.model_name}_{args.data_name}_{args.test_mode}_{started_at.tm_mday}{started_at.tm_hour}{started_at.tm_min}")  # , name=f"{config.session_name}_{started_at.tm_mday}D_{started_at.tm_hour}H_{started_at.tm_min}M")
    if args.tf32 == True:
        torch.backends.cuda.matmul.allow_tf32 = True
    Evaluator = Evaluator(args=args)
    Evaluator.evaluate()
    #print(f"{args.data_name},{args.model_name}ACC", Evaluation.evaluate())

    # wandb.finish()
