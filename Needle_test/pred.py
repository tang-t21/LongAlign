import yaml
import os
import glob
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from llama.modeling_llama import LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from torch import Tensor
from typing import List, Tuple, Any
import copy
# replace_llama_attn_with_flash_attn()
TRUNCATE_LEN = 128*1024

def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens
    return tok.decode(tokens, skip_special_tokens=True), len_after

def chunk_generate(
    model,
    tok,
    weight_percents,
    texts: List[str],
    max_tokens: int,
    sliding_window: int = 128 * 1024,
    chunk_size: int = 2500,
    temperature: float = 0.1,
    verbose: bool = False,
) -> List[str]:
    """
    Directly performing inference using HF transformers will result in OOM
    when using one A100 GPU. This is because the attention matrix is too large,
    so we chunk the input up and perform forward pass on each chunk to build
    up the KV cache. Note that each token still has to attend to
    all tokens in the past.
    """
    with torch.no_grad():
        """
        input_ids: (b, n)
        attention_mask: (b, n)
        [
            [0, 0, .., 0, 1, 1, ..., 1]
            ...
        ]
        """
        inputs = tok(texts, return_tensors="pt", truncation=False, add_special_tokens=False)
        inputs = inputs.to(model.device)  # type: ignore
        input_ids: Tensor = inputs.input_ids  # (b, n)
        needle_id = tok(" ★").input_ids[-1]
        num_needles = torch.sum(input_ids[0] == needle_id).item()
        print(f"Number of needles: {num_needles}")
        print(f"input_ids.shape:{input_ids.shape}")
        needle_positions = torch.where(input_ids[0] == needle_id)[0][:-1]
        print(f"needle_positions:{needle_positions}, {input_ids[0][needle_positions]}")
        attention_mask: Tensor = inputs.attention_mask  # (b, n)
        position_ids: Tensor = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, value=1)
        seq_len = input_ids.shape[-1]
        print("seq_len:", seq_len)
        kv_cache: Any = None
        # Split into chunks for pre-filling
        chunk_idxs = []
        n = seq_len - 1
        while n > 0:
            chunk_idxs.append(n)
            n -= chunk_size
        chunk_idxs.append(0)
        chunk_idxs = chunk_idxs[::-1]
        chunk_lo = chunk_idxs[:-1]
        chunk_hi = chunk_idxs[1:]
        print(f"Number of chunks: {len(chunk_lo)}, generating...")
        start_time = time.time()
        for chunk_i, (chunk_lo, chunk_hi) in enumerate(
            zip(chunk_lo, chunk_hi)
        ):
            if verbose:
                print(
                    f"[chunk {chunk_i}] {chunk_lo} : {chunk_hi}",
                    round(time.time() - start_time),
                )
            chunk_input_ids = input_ids[:, chunk_lo:chunk_hi]
            if kv_cache is not None:
                mask_start_idx = chunk_lo - kv_cache[0][0].shape[2]
            else:
                mask_start_idx = chunk_lo
            chunk_attention_mask = attention_mask[:, mask_start_idx:chunk_hi]
            chunk_position_ids = position_ids[:, chunk_lo:chunk_hi]
            outputs: BaseModelOutputWithPast = model.model.forward(
                input_ids=chunk_input_ids,
                attention_mask=chunk_attention_mask,
                position_ids=chunk_position_ids,
                past_key_values=kv_cache,
                return_dict=True,
                use_cache=True,
            )
            kv_cache = outputs.past_key_values
            # Discard KV states on the left beyond the window
            new_cache = ()
            n_layers = len(kv_cache)
            for layer_i in range(n_layers):
                keys = kv_cache[layer_i][0][:, :, -sliding_window:]
                values = kv_cache[layer_i][1][:, :, -sliding_window:]
                new_cache += ((keys, values),)
            kv_cache = new_cache
        kv_cache_len = kv_cache[0][0].shape[2]
        print("kv_cache_len:", kv_cache_len)
        print("KV cache prepared!")
        responses={}
        model.set_needle_positions(needle_positions.to(model.device))
        for weight_percent in weight_percents:
            model.set_weight_percent(weight_percent)
            outputs = model.generate(
                input_ids=input_ids[:, -1:],
                attention_mask=attention_mask[:, -kv_cache_len - 1 :],
                max_new_tokens=max_tokens,
                past_key_values=kv_cache,
                eos_token_id=tok.pad_token_id,
                use_cache=True,
                temperature=temperature,
                do_sample=False,
            )[0]
            responses[weight_percent]=tok.decode(outputs[1:], skip_special_tokens=True) # outputs[0] is input_ids[:, -1:] 
            print(f"response: {responses[weight_percent]}")
    return responses                                            

@torch.inference_mode()
def pred(model_name, model, tokenizer, weight_percents, messages, device, max_new_tokens=1024, temperature=0.1, verbose=False):
    # prompt = messages[0]['content']+'\n'+messages[1]['content']
    # prompt = tokenizer.apply_chat_template(messages)
    history = []
    if "internlm" in model_name or "chatglm" in model_name or "longalign-6b" in model_name:
        response, history = model.chat(tokenizer, prompt, history=history, max_new_tokens=max_new_tokens, temperature=temperature)
        return response
    elif "longalign-7b" in model_name or "longalign-13b" in model_name:
        if history == []:
            prompt = f"[INST]{prompt}[/INST]"
        else:
            prompt = history+"\n\n"+f"[INST]{prompt}[/INST]"
    elif "mistral" in model_name or "mixtral" in model_name:
        if history == []:
            prompt = f"<s>[INST] {prompt} [/INST]"
        else:
            prompt = history+f"</s> [INST] {prompt} [/INST]"
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama" in model_name:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # print("Truncating...")
    # prompt, context_length = truncate_by_tokens(prompt, tokenizer, TRUNCATE_LEN)
    if verbose:
        print("# chars:", len(prompt))
        print("=============== Input ===============")
        print(prompt[:200])
        print("...")
        print(prompt[-200:])
        print("=====================================")
    responses = chunk_generate(
        model,
        tokenizer,
        weight_percents,
        [prompt],
        max_tokens=max_new_tokens,
        chunk_size=1024,
        verbose=verbose,
        temperature=temperature,
    )
    # pred = tokenizer.decode(output, skip_special_tokens=True)
    return responses
    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
    context_length = input.input_ids.shape[-1]
    output = model.generate(
        **input,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        temperature=temperature,
    )[0]
    pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    return pred.strip()

def load_model_and_tokenizer(path, device):
    valid_path = path.lower()
    if "longchat" in valid_path or "vicuna" in valid_path:
        from fastchat.model import load_model
        model, _ = load_model(path, device='cpu', num_gpus=0, load_8bit=False, cpu_offloading=False, debug=False)
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    elif "mistral" in valid_path or "mixtral" in valid_path:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, use_flash_attention_2=True, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        model.generation_config = GenerationConfig.from_pretrained(path)
    elif "llama" in valid_path or "Llama" in valid_path:
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto", _attn_implementation="eager")
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    with open('/home/LongAlign/Needle_test/config-pred.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    model_provider = config['model']['model_provider']
    model_name = config['model']['model_name']
    prompt_dir = config['prompt_dir']
    save_dir = config['save_dir']
    max_new_tokens = config['max_new_tokens']
    weight_percents = config['weight_percents']
    context_lengths = config['context_lengths']
    num_needles = config['num_needles']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, tokenizer = load_model_and_tokenizer(model_name, device)
   
    for context_length in context_lengths:
        for filename in glob.glob(f'{prompt_dir}/{model_provider}_len_{context_length}_depth_3000_prompts.json'):
            print("Processing:", filename)
            with open(filename, 'r') as f:
                prompts = json.load(f)
            # print(filename)
            with open(f"/home/LongAlign/Needle_test/needle_weights.txt", "w") as file:
                file.write("\n")
            results = pred(model_name.lower(), model, tokenizer, weight_percents, prompts, device, max_new_tokens)
            # print(result)
            basename = os.path.basename(filename)
            newname = basename.replace('.json', '.txt').replace('_prompts', '')
            for weight_percent in weight_percents:
                result = results[weight_percent]
                with open(f'{save_dir}/{weight_percent}weight-{num_needles}needle-{newname}', 'w') as f:
                    f.write(result)


