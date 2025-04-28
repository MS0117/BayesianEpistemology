import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed, AutoModelForCausalLM, TextGenerationPipeline, \
    Text2TextGenerationPipeline
from datasets import load_dataset
from args import default_parse
import time
#import wandb
import argparse
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from peft import PeftModel, PeftConfig
from evaluate import load
import datasets
import openai
import tenacity
from tenacity import retry, wait_random_exponential
import json
from collections import OrderedDict
import os
from openai import OpenAI
import re
import random
from collections import Counter
from template import EXTRACT_ANSWER_FREQUENCY,VERBAL_LLAMA_TEMPLATE,VERBAL_GPT_TEMPLATE,VERBAL_PHI_TEMPLATE, TOKEN_LLAMA_TEMPLATE, TOKEN_GPT_TEMPLATE, TOKEN_PHI_TEMPLATE, SAMPLE_LLAMA_TEMPLATE,SAMPLE_GPT_TEMPLATE,SAMPLE_PHI_TEMPLATE,DIRECT_VERBAL_GPT_TEMPLATE,DIRECT_VERBAL_LLAMA_TEMPLATE,DIRECT_VERBAL_PHI_TEMPLATE,DIRECT_TOKEN_GPT_TEMPLATE,DIRECT_TOKEN_LLAMA_TEMPLATE,DIRECT_TOKEN_PHI_TEMPLATE,DIRECT_SAMPLE_LLAMA_TEMPLATE,DIRECT_SAMPLE_GPT_TEMPLATE,DIRECT_SAMPLE_PHI_TEMPLATE
with open('private_key.txt') as f:
    lines = f.readlines()
openai.api_key = lines[0].split('\n')[0]

client = OpenAI(
    api_key=openai.api_key,  # This is the default and can be omitted
)
#@retry(wait=wait_random_exponential(min=40, max=60), stop=tenacity.stop_after_attempt(10))
def Response(text, MODEL):
    # Question='Can you change the text style of Answer in the style writing of Response? The logical flow and final answer of Answer should be preserved.'
    if 'gpt' in MODEL.lower():

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": text},
            ],
            temperature=1,
            top_p=1.0,
            stop=["\"\"\""]
        )
        message = response.choices[0].message.content
        print("message",message)
    """
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
    """

    return message
@retry(wait=wait_random_exponential(min=40, max=60), stop=tenacity.stop_after_attempt(10))
def GeminiResponse(text, MODEL):
    # Question='Can you change the text style of Answer in the style writing of Response? The logical flow and final answer of Answer should be preserved.'
    if 'gemini' in MODEL.lower():
        # Initialize the Gemini model
        GOOGLE_API_KEY = gemini_api_key
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL)

        # Set generation configuration
        generation_config = {
            "candidate_count": 1,  # Single candidate
            "temperature": 1.0,
            "top_p": 1.0,

        }
       #print("gemini_response")

        # Generate the response
        response = model.generate_content(text, generation_config=generation_config)
        #print("response",response)
        # Extract the first candidate
        #candidate = response.result.candidates[0]  # Since candidate_count=1
        #print("candidate",candidate)
        # Extract text and log probability
        #text = candidate["content"]["parts"][0]["text"]
        print(response.text)



    return response.text


def Gemini_extract_response_text(response):

    try:
        if "Guess:" in response:
            return response.split("Guess: ")[1].split("\n")[0].strip()
        else:
            return response.split("\n")[0].strip()
    except:
        return ""


class Extractor(object):
    def __init__(self, args):
        self.args = args
        if 'llama' in self.args.model_name or 'gpt2' in self.args.model_name or 't5' in self.args.model_name or 'mistral' in self.args.model_name or 'phi' in self.args.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, cache_dir=self.args.cache_dir)
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.generation_kwargs = {"do_sample": True, "num_return_sequences": 1, "max_new_tokens": 512,
                                  "temperature": 1, "top_p":1}

    def prepare_dataset(self):

        test_data = []
        with open(self.args.test_set, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                test_data.append(ex)
        f.close()

        return test_data


    def prepare_model(self):
        if self.args.bf16:
            self.dtype = torch.bfloat16
        elif self.args.fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        try:
            model = AutoModelForCausalLM.from_pretrained(self.args.model_name, torch_dtype=self.dtype,
                                                         trust_remote_code=True, cache_dir=self.args.cache_dir,
                                                         low_cpu_mem_usage=True,attn_implementation="flash_attention_2")
        except:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name, torch_dtype=self.dtype,
                                                          trust_remote_code=True, cache_dir=self.args.cache_dir,
                                                          low_cpu_mem_usage=True,attn_implementation="flash_attention_2")

        return model

    def evaluate(self):
        directory_path = f"./sampling_extract/gemini/"

        # Create the directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)
        test_data = self.prepare_dataset()
        if self.args.api:
            file_name=self.args.model_name
        else:
            if 't5' in self.args.model_name:
                file_name = self.args.model_name[7:]
            elif 'llama' in self.args.model_name:
                file_name = self.args.model_name[11:]
            elif 'gpt2' in self.args.model_name:
                file_name = self.args.model_name
            elif 'phi' in self.args.model_name:
                file_name = self.args.model_name[10:]
            else:
                file_name = self.args.model_name
        data_name=self.args.test_set.split('/')[-1].split('.')[0]
        with open(f"{directory_path}{os.path.basename(self.args.test_set)}", "a", encoding="utf-8") as f:

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(device)


                if not self.args.api:
                    model = self.prepare_model()

                    if 'llama' in self.args.model_name or 'gpt2' in self.args.model_name or 'mistral' in self.args.model_name or 'phi' in self.args.model_name:
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

                for i in tqdm(range(0,len(test_data),10), total=len(test_data)):

                    if 'gpt' in self.args.model_name:
                        try:
                            response1 =test_data[i]["response"].split("Guess: ")[1].strip()
                            response2 =test_data[i+1]["response"].split("Guess: ")[1].strip()
                            response3 =test_data[i+2]["response"].split("Guess: ")[1].strip()
                            response4 =test_data[i+3]["response"].split("Guess: ")[1].strip()
                            response5 =test_data[i+4]["response"].split("Guess: ")[1].strip()
                            response6 =test_data[i+5]["response"].split("Guess: ")[1].strip()
                            response7 =test_data[i+6]["response"].split("Guess: ")[1].strip()
                            response8 =test_data[i+7]["response"].split("Guess: ")[1].strip()
                            response9 =test_data[i+8]["response"].split("Guess: ")[1].strip()
                            response10 =test_data[i+9]["response"].split("Guess: ")[1].strip()
                        except:
                            continue

                    elif 'gemini' in self.args.model_name:

                        response1 = Gemini_extract_response_text(test_data[i]["response"])
                        response2 = Gemini_extract_response_text(test_data[i+1]["response"])
                        response3 = Gemini_extract_response_text(test_data[i+2]["response"])
                        response4 = Gemini_extract_response_text(test_data[i+3]["response"])
                        response5 = Gemini_extract_response_text(test_data[i+4]["response"])
                        response6 = Gemini_extract_response_text(test_data[i+5]["response"])
                        response7 = Gemini_extract_response_text(test_data[i+6]["response"])
                        response8 = Gemini_extract_response_text(test_data[i+7]["response"])
                        response9 = Gemini_extract_response_text(test_data[i+8]["response"])
                        response10 = Gemini_extract_response_text(test_data[i+9]["response"])







                    input = EXTRACT_ANSWER_FREQUENCY.format(source1=response1,source2=response2,source3=response3,source4=response4,source5=response5,source6=response6,source7=response7,source8=response8,source9=response9,source10=response10)
                    print(input)
                    if self.args.api:
                        if 'gpt' in self.args.model_name:
                            response =Response(input, self.args.model_name)
                        elif 'gemini' in self.args.model_name:
                            response = Response(input, "gpt-4o-2024-05-13")


                    try:
                        guess_part, probability_part = response.split("\n")
                        guess = guess_part.split("Answer: ")[1].strip()
                        probability = float(probability_part.split("Frequency: ")[1].strip())

                        generated_answer = OrderedDict()

                        if 'SciQ' in self.args.test_set:
                            generated_answer["question"] = test_data[i]['question']
                            generated_answer["response"] = guess
                            generated_answer["probability"] = float(probability/10)
                            generated_answer["correct_answer"] = test_data[i]['correct_answer']
                        elif 'trivia' in self.args.test_set:
                            generated_answer["question"] = test_data[i]['question']
                            generated_answer["response"] = guess
                            generated_answer["probability"] = float(probability/10)
                            generated_answer["answers"] = test_data[i]['answers']
                        elif 'gsm' in self.args.test_set:
                            generated_answer["question"] = test_data[i]['question']
                            generated_answer["response"] = guess
                            generated_answer["probability"] = float(probability/10)
                            generated_answer["answers"] = test_data[i]['answers']


                        json.dump(generated_answer, f)  #
                        f.write("\n")
                    except:
                        continue






        f.close()


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

    #wandb.init(project=args.project,
    #           entity=args.entity,
    #           name=f"{args.model_name}_{args.data_name}_{started_at.tm_mday}{started_at.tm_hour}{started_at.tm_min}")  # , name=f"{config.session_name}_{started_at.tm_mday}D_{started_at.tm_hour}H_{started_at.tm_min}M")
    if args.tf32 == True:
        torch.backends.cuda.matmul.allow_tf32 = True
    Extractor = Extractor(args=args)
    Extractor.evaluate()
    # print(f"{args.data_name},{args.model_name}ACC", Evaluation.evaluate())

    # wandb.finish()
