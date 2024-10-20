from dotenv import load_dotenv
import os
import tiktoken
import glob
import json
import yaml
from anthropic import Anthropic
import numpy as np
import asyncio
from asyncio import Semaphore
from transformers import AutoTokenizer
import random
import string
load_dotenv()

SYSTEM_PROMPT = """You are a helpful AI bot that answers questions for a user. Keep your response short and direct"""

CONTEXT_PROMPT = """
<context>
{context}
</context>
"""

QUESTION_PROMPT = """
<question>
{question}
</question>
"""

HINT_PROMPT = """
Don't give information outside the document or repeat your findings.
"""

class Prompter:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 needles=["\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"],
                 haystack_dir="PaulGrahamEssays",
                 retrieval_question="What is the best thing to do in San Francisco?",
                 num_needles=4,
                 insert_type="depth_percent",
                 context_lengths_min = 1000,
                 context_lengths_max = 200000,
                 context_lengths_num_intervals = 35,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 35,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 tokenizer_type = "OpenAI",
                 model_name = "gpt-4-1106-preview",
                 num_concurrent_requests = 1,
                 final_context_length_buffer = 0,
                 save_dir = "prompts",
                 print_ongoing_status = True):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needles or not haystack_dir or not retrieval_question or not num_needles:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        self.needles = needles
        self.haystack_dir = haystack_dir
        self.num_needles = num_needles
        self.retrieval_question = retrieval_question
        self.num_concurrent_requests = num_concurrent_requests
        self.final_context_length_buffer = final_context_length_buffer
        self.print_ongoing_status = print_ongoing_status
        self.tokenizer_type = tokenizer_type
        self.model_name = model_name
        self.testing_results = []
        self.insert_type = insert_type
        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals+1, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals+1, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals+1)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        if self.tokenizer_type == "OpenAI":
            assert self.model_name is not None, "If you're using OpenAI, you must provide a model name."
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        elif self.tokenizer_type == "Anthropic":
            self.tokenizer = Anthropic().get_tokenizer()
        elif self.tokenizer_type == "Huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, use_fast=False)
        else:
            raise ValueError("tokenizer_type is not supported. Must be either 'tiktoken', 'Anthropic', or 'Huggingface'")
        
        self.save_dir = save_dir

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    
    async def bound_evaluate_and_log(self, sem, *args):
        async with sem:
            await self.evaluate_and_log(*args)

    async def run_test(self):
        sem = Semaphore(self.num_concurrent_requests)

        # Run through each iteration of context_lengths and depths
        tasks = []
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(sem, context_length, depth_percent, self.num_needles)
                tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    def generate_prompt(self, context, retrieval_question):
        prompt_format = [{
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": CONTEXT_PROMPT.format(context=context) + "\n\n" + QUESTION_PROMPT.format(question=retrieval_question) + "\n\n" + HINT_PROMPT
            }]

        if not self.tokenizer.chat_template:
            self.tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        
        if "system" not in self.tokenizer.chat_template or "raise_exception('System role not supported')" in self.tokenizer.chat_template:
            prompt_format.pop(0)
            prompt_format[0]["content"] = SYSTEM_PROMPT + "\n" + prompt_format[0]["content"]

        # return self.tokenizer.apply_chat_template(prompt_format, tokenize=False, add_generation_prompt=True)
        return prompt_format

    async def evaluate_and_log(self, context_length, depth_percent, num_needles):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later

        # Go generate the required length context and place your needle statement in
        context = await self.generate_context(context_length, depth_percent, num_needles)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.generate_prompt(context, self.retrieval_question)

        context_file_location = f'{self.tokenizer_type}_len_{context_length}_depth_{int(depth_percent*100)}'

        # Save the prompts to file for retesting
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Save the result to file for retesting
        with open(f'{self.save_dir}/{context_file_location}_prompts.json', 'w', encoding="utf-8") as f:
            json.dump(prompt, f, ensure_ascii=False)

    async def generate_context(self, context_length, depth_percent, num_needles):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needles(context, depth_percent, context_length, num_needles)

        return context
    
    def encode_text_to_tokens(self, text, no_bos=False):
        if self.tokenizer_type == "OpenAI":
            return self.tokenizer.encode(text)
        elif self.tokenizer_type == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.tokenizer.encode(text).ids
        elif self.tokenizer_type == "Huggingface":
            tokens=self.tokenizer(text, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids.view(-1).tolist()
            return tokens[1:] if no_bos else tokens
        else:
            raise ValueError("tokenizer_type must be either 'OpenAI', 'Anthropic', or 'Huggingface'")
    
    def insert_needles(self, context, depth_percent, context_length, num_needles):
        """
        Inserts multiple needles (specific facts or pieces of information) into the original context string at 
        designated depth percentages, effectively distributing these needles throughout the context. This method 
        is designed to test a model's ability to retrieve specific information (needles) from a larger body of text 
        (haystack) based on the placement depth of these needles.

        The method first encodes the context and each needle into tokens to calculate their lengths in tokens. 
        It then adjusts the context length to accommodate the final buffer length. This is crucial for ensuring 
        that the total token count (context plus needles) does not exceed the maximum allowable context length, 
        which might otherwise lead to information being truncated.

        This approach calculates the initial insertion point for the first needle as before but then calculates even 
        spacing for the remaining needles based on the remaining context length. It ensures that needles are 
        distributed as evenly as possible throughout the context after the first insertion. 
        
        Args:
            context (str): The original context string.
            depth_percent (float): The depth percent at which to insert the needles.
            context_length (int): The total length of the context in tokens, adjusted for final buffer.
        
        Returns:
            str: The new context with needles inserted.
        """
        tokens_context = self.encode_text_to_tokens(context, no_bos=True)
        context_length -= self.final_context_length_buffer
        # with open(f'./needles/{num_needles}keys_len_4.json', 'r') as f:
        #     data = json.load(f)
        #     self.needles = data['needles']
        #     self.retrieval_question = data['retrieval_question']
        self.needles = [" ★ "] * num_needles
        self.retrieval_question = "How many ★ are there in the above context? Just provide a number as the answer."
        # Calculate the total length of all needles in tokens
        total_needles_length = sum(len(self.encode_text_to_tokens(needle)) for needle in self.needles)
        # Ensure context length accounts for needles
        if len(tokens_context) + total_needles_length > context_length:
            tokens_context = tokens_context[:context_length - total_needles_length]
        
        # Insert the needles with 'depth_percent' method
        if self.insert_type == "depth_percent":
            # To evenly distribute the needles, we calculate the intervals they need to be inserted.
            depth_percent_interval = (100 - depth_percent) / len(self.needles)
            
            # Reset the insertion percentages list for the current context
            self.insertion_percentages = []

        # Insert the needles with 'random' method
        elif self.insert_type == "random":
            # Reset the insertion percentages list for the random insertion points
            self.insertion_percentages = []

            # To randomly and evenly distribute the needles, we calculate the insertion range and range intervals.
            range_interval = int(100 / num_needles)

            # Generate random insertion points for each needle based on the range interval.
            for i in range(num_needles):
                self.insertion_percentages.append(random.randrange(i * range_interval, (i + 1) * range_interval))

        # Insert needles at calculated points
        insertion_positions = []
        for i in range(len(self.needles)):
            tokens_needle = self.encode_text_to_tokens(self.needles[i])

            if depth_percent == 100:
                # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
                tokens_context = tokens_context + tokens_needle
            else:
                # Go get the position (in terms of tokens) to insert your needle
                if self.insert_type == "random":
                    insertion_point = int(len(tokens_context) * (self.insertion_percentages[i] / 100))    
                elif self.insert_type == "depth_percent":
                    insertion_point = int(len(tokens_context) * (depth_percent / 100))

                # tokens_new_context represents the tokens before the needle
                tokens_new_context = tokens_context[:insertion_point]

                # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
                period_tokens = self.encode_text_to_tokens('.')
                
                # Then we iteration backwards until we find the first period
                while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                    insertion_point -= 1
                    tokens_new_context = tokens_context[:insertion_point]
                insertion_positions.append(insertion_point)
                # Insert the needle into the context at the found position
                tokens_context = tokens_context[:insertion_point] + tokens_needle + tokens_context[insertion_point:]

                # Log 
                insertion_percentage = (insertion_point / len(tokens_context)) * 100
                
                if self.insert_type == "random":
                    self.insertion_percentages[i] = insertion_percentage

                if self.insert_type == "depth_percent":
                    self.insertion_percentages.append(insertion_percentage)
                    # Adjust depth for next needle
                    depth_percent += depth_percent_interval

                print(f"Inserted '{self.needles[i]}' at {insertion_percentage:.2f}% of the context, total length now: {len(tokens_context)} tokens")
        print(f"Needles inserted at positions: {insertion_positions}")
        new_context = self.decode_tokens(tokens_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        if self.tokenizer_type == "OpenAI":
            return len(self.tokenizer.encode(context))
        elif self.tokenizer_type == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return len(self.tokenizer.encode(context).ids)
        elif self.tokenizer_type == "Huggingface":
            return self.tokenizer(context, truncation=False, return_tensors="pt").input_ids.shape[-1]
        else:
            raise ValueError("tokenizer_type must be either 'OpenAI', 'Anthropic', or 'Huggingface'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)
        files = glob.glob(f"{self.haystack_dir}/*.txt")
        random.seed(114514)
        random.shuffle(files)
        while self.get_context_length_in_tokens(context) < max_context_length:
            file = random.choice(files)
            with open(file, 'r') as f:
                context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.tokenizer_type == "OpenAI":
            return self.tokenizer.encode(context)
        elif self.tokenizer_type == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.tokenizer.encode(context).ids
        elif self.tokenizer_type == "Huggingface":
            return self.tokenizer(context, truncation=False, return_tensors="pt").input_ids.view(-1).tolist()
        else:
            raise ValueError("tokenizer_type must be either 'OpenAI', 'Anthropic', or 'Huggingface'")
        
    def decode_tokens(self, tokens, context_length=None):
        if self.tokenizer_type == "OpenAI":
            return self.tokenizer.decode(tokens[:context_length])
        elif self.tokenizer_type == "Anthropic":
            # Assuming you have a different decoder for Anthropic
            return self.tokenizer.decode(tokens[:context_length])
        elif self.tokenizer_type == "Huggingface":
            decoded = self.tokenizer.decode(tokens[:context_length], skip_special_tokens=True)
            return decoded
        else:
            raise ValueError("tokenizer_type must be either 'OpenAI', 'Anthropic', or 'Huggingface'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Prompt Generation ...")
        print (f"- Tokenizer: {self.tokenizer_type}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {[needle.strip() for needle in self.needles]}")
        print ("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        asyncio.run(self.run_test())


if __name__ == '__main__':
    with open('config-prompt.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    ht = Prompter(
        needles=config['prompt']['needles'],
        haystack_dir=config['prompt']['haystack_dir'],
        retrieval_question=config['prompt']['retrieval_question'],
        num_needles=config['prompt']['num_needles'],
        context_lengths_min=config['context']['min_len'],
        context_lengths_max=config['context']['max_len'],
        context_lengths_num_intervals=config['context']['interval'],
        context_lengths=config['context']['manually_select_list'],

        document_depth_percent_min=config['document_depth']['min_percent'],
        document_depth_percent_max=config['document_depth']['max_percent'],
        document_depth_percent_intervals=config['document_depth']['interval'],
        document_depth_percents=config['document_depth']['manually_select_list'],
        document_depth_percent_interval_type=config['document_depth']['interval_type'],

        tokenizer_type=config['tokenizer']['tokenizer_type'],
        model_name=config['tokenizer']['model_name'],
        final_context_length_buffer=config['context']['final_context_length_buffer'],
        save_dir=config['save_dir'],
    )
    # tokens = ht.encode_text_to_tokens("Beijing is the capital of China.")
    # print(tokens)
    # print(ht.decode_tokens(tokens))

    # tokens = ht.encode_text_to_tokens(" ★")
    # print(len(tokens))
    # print(tokens)
    # print(ht.decode_tokens([271]))
    ht.start_test()
