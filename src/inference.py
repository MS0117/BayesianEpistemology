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
import re
import random
from collections import Counter
from template import VERBAL_LLAMA_TEMPLATE,VERBAL_GPT_TEMPLATE,VERBAL_PHI_TEMPLATE, TOKEN_LLAMA_TEMPLATE, TOKEN_GPT_TEMPLATE, TOKEN_PHI_TEMPLATE, SAMPLE_LLAMA_TEMPLATE,SAMPLE_GPT_TEMPLATE,SAMPLE_PHI_TEMPLATE,DIRECT_VERBAL_GPT_TEMPLATE,DIRECT_VERBAL_LLAMA_TEMPLATE,DIRECT_VERBAL_PHI_TEMPLATE,DIRECT_TOKEN_GPT_TEMPLATE,DIRECT_TOKEN_LLAMA_TEMPLATE,DIRECT_TOKEN_PHI_TEMPLATE,DIRECT_SAMPLE_LLAMA_TEMPLATE,DIRECT_SAMPLE_GPT_TEMPLATE,DIRECT_SAMPLE_PHI_TEMPLATE
with open('key.txt') as f:
    lines = f.readlines()
openai.api_key = lines[0].split('\n')[0]

with open('key_gemini.txt') as f:
    lines = f.readlines()
gemini_api_key = lines[0].split('\n')[0]

import google.generativeai as genai



@retry(wait=wait_random_exponential(min=40, max=60), stop=tenacity.stop_after_attempt(10))
def Response(text, MODEL):
    # Question='Can you change the text style of Answer in the style writing of Response? The logical flow and final answer of Answer should be preserved.'
    if 'gpt' in MODEL.lower():
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": text},
            ],
            temperature=1,
            top_p=1.0,
            stop=["\"\"\""]
        )
        message = response["choices"][0]["message"]["content"]
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
def LogProbResponse(text, MODEL):
    # Question='Can you change the text style of Answer in the style writing of Response? The logical flow and final answer of Answer should be preserved.'
    if 'gpt' in MODEL.lower():
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": text},
            ],
            temperature=1,
            top_p=1.0,
            logprobs=True,
            stop=["\"\"\""]
        )
        #print(response)
        message = response["choices"][0]["message"]["content"]
        log_prob= sum([item["logprob"] for item in response["choices"][0]["logprobs"]["content"]])

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

    return message,log_prob







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


@retry(wait=wait_random_exponential(min=40, max=60), stop=tenacity.stop_after_attempt(10))
def GeminiLogProbResponse(text, MODEL):
    """
    Generate text using the Gemini API with a single candidate.

    Args:
        prompt (str): The input prompt for text generation.
        model_name (str): The Gemini model to use (default: 'gemini-1.5-flash').

    Returns:
        dict: A dictionary containing the generated text, avg_logprobs, and probability.
    """
    # Initialize the Gemini model

    if 'gemini' in MODEL.lower():
        GOOGLE_API_KEY = gemini_api_key
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL)

        # Set generation configuration
        generation_config = {
            "candidate_count": 1,  # Single candidate
            "temperature": 1.0,
            "top_p": 1.0,

        }

        # Generate the response
        response = model.generate_content(text, generation_config=generation_config)
        #print("gemini_response")
        #print(response)
        # Extract the first candidate
        #candidate = response.result.candidates  # Since candidate_count=1
        #print("candidate:",candidate)
        # Extract text and log probability
        text = response.text
        avg_logprobs = response.candidates[0].avg_logprobs

        # Convert log probability to 0-1 scale
        print("text",text)
        print(avg_logprobs)
        #print("log_prob",avg_logprobs)


        # Return the results


    return text,avg_logprobs








class Evaluator(object):
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
        directory_path = f"./11_{started_at.tm_mday}/"

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
        with open(f"{directory_path}{file_name}_{data_name}_{self.args.confidence_method}_{self.args.answer_strategy}.jsonl", "a", encoding="utf-8") as f:


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

                for i in tqdm(range(len(test_data)), total=len(test_data)):
                    if 'verbal' in self.args.confidence_method:
                        if 'llama' in args.model_name.lower():
                            if 'evidence' in self.args.answer_strategy:
                                if 'coincidence' in self.args.test_set:
                                    input = VERBAL_LLAMA_TEMPLATE.format(question=test_data[i]['question'],evidence=test_data[i]['coincidental support'])

                                else:
                                    input = VERBAL_LLAMA_TEMPLATE.format(question=test_data[i]['question'],evidence=test_data[i]['support'])
                            else:
                                input = DIRECT_VERBAL_LLAMA_TEMPLATE.format(question=test_data[i]['question'])
                        if 'gpt' in self.args.model_name.lower() or 'gemini' in args.model_name.lower():
                            print("gemini")
                            if 'evidence' in self.args.answer_strategy:
                                if 'coincidence' in self.args.test_set:
                                    input = VERBAL_GPT_TEMPLATE.format(question=test_data[i]['question'],  evidence=test_data[i]['coincidental support'])

                                else:
                                    input = VERBAL_GPT_TEMPLATE.format(question=test_data[i]['question'],evidence=test_data[i]['support'])
                            else:
                                input = DIRECT_VERBAL_GPT_TEMPLATE.format(question=test_data[i]['question'])

                        if 'phi' in args.model_name.lower():
                            if 'evidence' in self.args.answer_strategy:
                                if 'coincidence' in self.args.test_set:
                                    input = VERBAL_PHI_TEMPLATE.format(question=test_data[i]['question'],
                                                                  evidence=test_data[i]['coincidental support'])

                                else:
                                    input = VERBAL_PHI_TEMPLATE.format(question=test_data[i]['question'],
                                                                  evidence=test_data[i]['support'])
                            else:
                                input = DIRECT_VERBAL_PHI_TEMPLATE.format(question=test_data[i]['question'])
                        #print(input)
                        if self.args.api:
                            if 'gpt' in self.args.model_name.lower():
                                response =Response(input, self.args.model_name)
                            elif 'gemini' in self. args.model_name.lower():
                                print("gemini")
                                response=GeminiResponse(input, self.args.model_name)



                        else:
                            response = generate_text(input)

                        generated_answer = OrderedDict()

                        if 'SciQ' in self.args.test_set:
                            generated_answer["question"] = test_data[i]['question']
                            generated_answer["response"] = response
                            generated_answer["correct_answer"]= test_data[i]['correct_answer']
                        elif 'trivia'   in self.args.test_set:
                            generated_answer["question"] = test_data[i]['question']
                            generated_answer["response"] = response
                            generated_answer["answers"]= test_data[i]['answers']


                        elif 'gsm' in self.args.test_set:
                            generated_answer["question"] = test_data[i]['question']
                            generated_answer["response"] = response
                            generated_answer["answers"] = test_data[i]['answer']
                        json.dump(generated_answer, f)  #
                        f.write("\n")


                    if 'token_prob' in self.args.confidence_method:
                        if 'llama' in args.model_name.lower():
                            if 'evidence' in self.args.answer_strategy:
                                if 'coincidence' in self.args.test_set:
                                    input = TOKEN_LLAMA_TEMPLATE.format(question=test_data[i]['question'],
                                                                  evidence=test_data[i]['coincidental support'])

                                else:
                                    input = TOKEN_LLAMA_TEMPLATE.format(question=test_data[i]['question'],
                                                                  evidence=test_data[i]['support'])

                            else:
                                input = DIRECT_TOKEN_LLAMA_TEMPLATE.format(question=test_data[i]['question'])
                        if 'gpt' in args.model_name.lower() or 'gemini' in args.model_name.lower():
                            print("gemini")
                            if 'evidence' in self.args.answer_strategy:
                                if 'coincidence' in self.args.test_set:
                                    input = TOKEN_GPT_TEMPLATE.format(question=test_data[i]['question'],
                                                                  evidence=test_data[i]['coincidental support'])

                                else:
                                    input = TOKEN_GPT_TEMPLATE.format(question=test_data[i]['question'],
                                                                  evidence=test_data[i]['support'])
                            else:
                                input = DIRECT_TOKEN_GPT_TEMPLATE.format(question=test_data[i]['question'])
                        if 'phi' in args.model_name.lower():
                            if 'evidence' in self.args.answer_strategy:
                                if 'coincidence' in self.args.test_set:
                                    input = TOKEN_PHI_TEMPLATE.format(question=test_data[i]['question'],
                                                                  evidence=test_data[i]['coincidental support'])

                                else:
                                    input = TOKEN_PHI_TEMPLATE.format(question=test_data[i]['question'],
                                                                  evidence=test_data[i]['support'])

                            else:
                                input = DIRECT_TOKEN_PHI_TEMPLATE.format(question=test_data[i]['question'])

                        if self.args.api:
                            if 'gpt' in self.args.model_name.lower():
                                response,log_prob=LogProbResponse(input, self.args.model_name)
                            elif 'gemini' in self.args.model_name.lower():
                                print("gemini")
                                response,log_prob= GeminiLogProbResponse(input, self.args.model_name)


                        else:
                            input_ids = self.tokenizer(input, return_tensors="pt").input_ids.to(device)
                            outputs = model.generate(input_ids, max_new_tokens=256,
                                                              #  num_return_sequences=self.num_samples, do_sample=True,
                                                              return_dict_in_generate=True, output_scores=True)
                            # num_beams=5)
                            transition_scores = language_model.compute_transition_scores(
                                outputs.sequences, outputs.scores, normalize_logits=True
                                # NOTE normalize SHOULD be true?
                            )
                            output_length = input_ids.shape[1] + np.sum(transition_scores.cpu().numpy() < 0, axis=1)
                            length_penalty = language_model.generation_config.length_penalty
                            confidence = torch.exp(
                                transition_scores.cpu().sum(axis=1) / (output_length ** length_penalty))
                            response = self.tokenizer.batch_decode(outputs.sequences.cpu(), skip_special_tokens=True)

                        generated_answer = OrderedDict()
                        if 'SciQ' in self.args.test_set:
                            generated_answer["question"] = test_data[i]['question']
                            generated_answer["response"] = response
                            generated_answer["log_prob"]=log_prob
                            generated_answer["linearized_prob"]=np.round(np.exp(log_prob)*100,2)
                            generated_answer["correct_answer"] = test_data[i]['correct_answer']

                        elif 'trivia' in self.args.test_set:
                            generated_answer["question"] = test_data[i]['question']
                            generated_answer["response"] = response
                            generated_answer["log_prob"]=log_prob
                            generated_answer["linearized_prob"]=np.round(np.exp(log_prob)*100,2)
                            generated_answer["answers"] = test_data[i]['answers']


                        elif 'gsm' in self.args.test_set:
                            generated_answer["question"] = test_data[i]['question']
                            generated_answer["response"] = response
                            generated_answer["log_prob"]=log_prob
                            generated_answer["linearized_prob"]=np.round(np.exp(log_prob)*100,2)
                            generated_answer["answers"] = test_data[i]['answer']

                        json.dump(generated_answer, f)  #
                        f.write("\n")

                    if 'sampling' in self.args.confidence_method:
                        for sam_num in range(self.args.num_sample):
                            if 'llama' in args.model_name.lower():
                                if 'evidence' in self.args.answer_strategy:
                                    if 'coincidence' in self.args.test_set:
                                        input = SAMPLE_LLAMA_TEMPLATE.format(question=test_data[i]['question'],
                                                                      evidence=test_data[i]['coincidental support'])

                                    else:
                                        input = SAMPLE_LLAMA_TEMPLATE.format(question=test_data[i]['question'],
                                                                      evidence=test_data[i]['support'])
                                else:
                                    input = DIRECT_SAMPLE_LLAMA_TEMPLATE.format(question=test_data[i]['question'])


                            if 'gpt' in args.model_name.lower() or  'gemini' in args.model_name.lower():
                                if 'evidence' in self.args.answer_strategy:
                                    if 'coincidence' in self.args.test_set:
                                        input = SAMPLE_GPT_TEMPLATE.format(question=test_data[i]['question'],
                                                                      evidence=test_data[i]['coincidental support'])

                                    else:
                                        input = SAMPLE_GPT_TEMPLATE.format(question=test_data[i]['question'],
                                                                      evidence=test_data[i]['support'])
                                else:
                                    input = DIRECT_SAMPLE_GPT_TEMPLATE.format(question=test_data[i]['question'])
                            if 'phi' in args.model_name.lower():
                                if 'evidence' in self.args.answer_strategy:
                                    if 'coincidence' in self.args.test_set:
                                        input = SAMPLE_PHI_TEMPLATE.format(question=test_data[i]['question'],
                                                                      evidence=test_data[i]['coincidental support'])

                                    else:
                                        input = SAMPLE_PHI_TEMPLATE.format(question=test_data[i]['question'],
                                                                      evidence=test_data[i]['support'])
                                else:
                                    input = DIRECT_SAMPLE_PHI_TEMPLATE.format(question=test_data[i]['question'])

                            if self.args.api:
                                if 'gpt'in self.args.model_name:
                                    response =Response(input, self.args.model_name)
                                elif 'gemini' in self.args.model_name:
                                    response = GeminiResponse(input, self.args.model_name)


                            else:
                                response = generate_text(input)


                            generated_answer = OrderedDict()
                            if 'SciQ' in self.args.test_set:
                                generated_answer["question"] = test_data[i]['question']
                                generated_answer["sample_num"]=sam_num
                                generated_answer["response"] = response
                                generated_answer["correct_answer"] = test_data[i]['correct_answer']


                            elif 'trivia' in self.args.test_set:
                                generated_answer["question"] = test_data[i]['question']
                                generated_answer["sample_num"] = sam_num
                                generated_answer["response"] = response
                                generated_answer["answers"] = test_data[i]['answers']



                            elif 'gsm' in self.args.test_set:
                                generated_answer["question"] = test_data[i]['question']
                                generated_answer["sample_num"] = sam_num
                                generated_answer["response"] = response
                                generated_answer["answers"] = test_data[i]['answer']

                            json.dump(generated_answer, f)  #
                            f.write("\n")


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
    #          name=f"{args.model_name}_{args.data_name}_{started_at.tm_mday}{started_at.tm_hour}{started_at.tm_min}")  # , name=f"{config.session_name}_{started_at.tm_mday}D_{started_at.tm_hour}H_{started_at.tm_min}M")
    if args.tf32 == True:
        torch.backends.cuda.matmul.allow_tf32 = True
    Evaluator = Evaluator(args=args)
    Evaluator.evaluate()
    # print(f"{args.data_name},{args.model_name}ACC", Evaluation.evaluate())

    # wandb.finish()
