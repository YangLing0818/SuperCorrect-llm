import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_loader import load_data
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import parse_question,parse_ground_truth
from model_utils import generate_completions
from llm_evaluation import DataProcessor
import json
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="math", type=str)
    parser.add_argument("--data_dir", default="./dataset", type=str)
    parser.add_argument("--model_name_or_path", default="models/SuperCorrect-7B", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="hierarchical", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    parser.add_argument("--api_key",type=str,default='sk-hHFKw8usRH7nE2cq63EfCe81E0Ef490aAf97D25917697d08',help="The api_key to enable LLM function as evaluator")
    parser.add_argument("--num_processes",type=int,default=16,help="Number of processes when evaluating results")
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args

def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file

def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if args.use_vllm:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
        )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype = "auto",
            device_map = "auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,padding_side="left")
            # set pad token to eos token if pad token is not set
        if tokenizer.pad_token is None:
            if tokenizer.unk_token:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.unk_token_id
            elif tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                raise ValueError("You are using a new tokenizer without a pad token."
                                "This is not supported by this script.")
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        main(llm, tokenizer, data_name, args)

    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))
def main(llm, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])
    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)
        if idx == args.start:
            print(full_prompt)
        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }
        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)
            # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    start_time = time.time()
    print("-" * 20)
    current_prompts = remain_prompts
    # get all outputs
    prompts = [item[1] for item in current_prompts]
    if args.use_vllm:
        outputs = llm.generate(
            prompts,
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                n=1,
                stop=stop_words,
                stop_token_ids=(
                    [151645, 151643]
                    if "qwen2" in args.model_name_or_path.lower()
                    else None
                ),
            ),
        )
        outputs = sorted(
            outputs, key=lambda x: int(x.request_id)
        )  # sort outputs by request_id
        outputs = [output.outputs[0].text for output in outputs]
    else:
        outputs = generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
            )
    assert len(outputs) == len(current_prompts)
    for i, sample in enumerate(samples):
        sample['respond'] = outputs[i]
    processor = DataProcessor(api_key=args.api_key)
    samples = processor.process_data_in_parallel(samples, args.num_processes)
    save_jsonl(samples, out_file)
    correct = 0
    for sample in samples:
        if sample['correct'] == True:
            correct += 1
    acc = correct/len(samples)
    print(f'Accuracy:{acc}')
    results = {'num_samples:':len(samples),'acc:':acc}
    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(results, f, indent=4)
if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
