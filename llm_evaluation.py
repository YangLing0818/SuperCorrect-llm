import json
import logging
import multiprocessing
from tqdm import tqdm
from openai import OpenAI
import os
import argparse
import time
from pydantic import BaseModel
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
import functools

# Structured output
class Extractor(BaseModel):
    respond_answer: str
    correct: bool
    
class DataProcessor:
    def __init__(self, api_key, base_url="https://api.openai.com/v1/"):
        self.api_key = api_key
        self.base_url = base_url

    def get_response(self, client, prompt, problem):
        try:
            # GET OPENAI respond
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": problem},
                ],
                response_format=Extractor,
            )
            event = completion.choices[0].message.parsed
            return event
        except Exception as e:
            logging.error(f"Error in get_response: {e}")
            return None

    def process_item(self, item):
        try:
            # Initialize client
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            
            # Construct problems
            problem = 'Ground_Truth:' + item['gt'] + '\n' + 'XML_response' + item['respond']
            prompt = """You will be given a ground truth answer and a respond in XML format from LLM. Your task is to extract the answer in XML respond and judge if the XML answer is correct consider the ground truth answer."""

            # GET response
            response = self.get_response(client, prompt, problem)
            if response is not None:
                item['respond_answer'] = response.respond_answer
                item['correct'] = response.correct
            else:
                item['respond_answer'] = None
                item['correct'] = None
        except Exception as e:
            logging.error(f"Error processing item: {e}")
        return item

    def process_data_in_parallel(self, data, num_processes):
        # Multi-processing 
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(self.process_item, data), total=len(data)))
        return results


