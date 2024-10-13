## SuperCorrect: Supervising and Correcting Language Models with Error-Driven Insights

> [**SuperCorrect: Supervising and Correcting Language Models with Error-Driven Insights**](link)  
> [Ling Yang\*](https://yangling0818.github.io/), [Zhaochen Yu*](https://github.com/BitCodingWalkin), [Tianjun Zhang](https://tianjunz.github.io/), [Minkai Xu](https://minkaixu.com/), [Joseph E. Gonzalez](https://people.eecs.berkeley.edu/~jegonzal/),[Bin Cui](https://cuibinpku.github.io/), [Shuicheng Yan](https://yanshuicheng.info/)  
> Peking University, Skywork AI, UC Berkeley, Stanford University 

<p align="left">
  <a href='https://arxiv.org/abs/to be edited'>
  <img src='https://img.shields.io/badge/Arxiv-2410.07171-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
  <a href='https://huggingface.co/BitStarWalkin/SuperCorrect-7B'>
  <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow'></a>
  </p>
<details>
    <summary>Click for full abstract</summary>
    Large language models (LLMs) like GPT-4, PaLM, and LLaMA have shown significant improvements in various reasoning tasks. However, smaller models such as Llama-3-8B and DeepSeekMath-Base still struggle with complex mathematical reasoning because they fail to effectively identify and correct reasoning errors. Recent reflection-based methods aim to address these issues by enabling self-reflection and self-correction, but they still face challenges in independently detecting errors in their reasoning steps.
    To overcome these limitations, we propose **SuperCorrect**, a novel two-stage framework that uses a large teacher model to *supervise* and *correct* both the reasoning and reflection processes of a smaller student model. In the first stage, we extract hierarchical high-level and detailed thought templates from the teacher model to guide the student model in eliciting more fine-grained reasoning thoughts. In the second stage, we introduce cross-model collaborative direct preference optimization (DPO) to enhance the self-correction abilities of the student model by following the teacher's correction traces during training. This cross-model DPO approach teaches the student model to effectively locate and resolve erroneous thoughts with error-driven insights from the teacher model, breaking the bottleneck of its thoughts and acquiring new skills and knowledge to tackle challenging problems. 
    Extensive experiments consistently demonstrate our superiority over previous methods. Notably, our **SuperCorrect-7B** model significantly **surpasses powerful DeepSeekMath-7B by 7.8%/5.3% and Qwen2.5-Math-7B by 15.1%/6.3%** on MATH/GSM8K benchmarks, achieving new SOTA performance among all 7B models.
</details>

## Introduction

![image](imgs/intro.png)

This repo provides the official implementation of SuperCorrect,  a novel two-stage fine-tuning method for improving both reasoning accuracy and self-correction ability for LLMs. Notably, our **SupperCorrect-7B** model significantly surpasses powerful **DeepSeekMath-7B by 7.8%/5.3% and Qwen2.5-Math-7B by 15.1%/6.3% on MATH/GSM8K benchmarks**, achieving new SOTA performance among all 7B models.



🚨 Unlike other LLMs, we incorporate LLMs with our pre-defined hierarchical thought template ([Buffer of Thought (BoT)](https://github.com/YangLing0818/buffer-of-thought-llm)) to conduct more deliberate reasoning than conventional CoT. It should be noted that our evaluation methods relies on pure mathematical reasoning abilities of LLMs, instead of leverage other programming methods such as PoT and ToRA.


## Examples

![image](imgs/example1.png)

<div align="left">
<b>
🚨 For more concise and clear presentation, we omit some XML tags  
</b>
</div>

## Quick Start

### Installation

```bash
git clone https://github.com/YangLing0818/SuperCorrect
cd SuperCorrect
conda create -n SuperCorrect python==3.10
conda activate SuperCorrect
pip install -r requirements.txt
```

### Requirements

* Since our current model is  based on Qwen2.5-Math series, `transformers>=4.37.0` is needed for Qwen2.5-Math models. The latest version is recommended.

> [!Warning]
>
> <div align="center">
> <b>
> 🚨 This is a must because `transformers` integrated Qwen2 codes since `4.37.0`.
> </b>
> </div>

### Inference with Different Library

#### 🤗 Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "BitStarWalkin/SuperCorrect-7B"
device = "cuda" 

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Find the distance between the foci of the ellipse \[9x^2 + \frac{y^2}{9} = 99.\]"
hierarchical_prompt = "Solve the following math problem in a step-by-step XML format, each step should be enclosed within tags like <Step1></Step1>. For each step enclosed within the tags, determine if this step is challenging and tricky, if so, add detailed explanation and analysis enclosed within <Key> </Key> in this step, as helpful annotations to help you thinking and remind yourself how to conduct reasoning correctly. After all the reasoning steps, summarize the common solution and reasoning steps to help you and your classmates who are not good at math generalize to similar problems within <Generalized></Generalized>. Finally present the final answer within <Answer> </Answer>."
# HT
messages = [
    {"role": "system", "content":hierarchical_prompt },
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

#### 🔥 vLLM

```python
import os
from vllm import LLM, SamplingParams
model_name = 'BitStarWalkin/SuperCorrect-7B'
hierarchical_prompt = "Solve the following math problem in a step-by-step XML format, each step should be enclosed within tags like <Step1></Step1>. For each step enclosed within the tags, determine if this step is challenging and tricky, if so, add detailed explanation and analysis enclosed within <Key> </Key> in this step, as helpful annotations to help you thinking and remind yourself how to conduct reasoning correctly. After all the reasoning steps, summarize the common solution and reasoning steps to help you and your classmates who are not good at math generalize to similar problems within <Generalized></Generalized>. Finally present the final answer within <Answer> </Answer>."
prompts = [
    "For what positive value of $t$ is $|{-4+ti}| = 6$?",
    "Find the distance between the foci of the ellipse \\[9x^2 + \\frac{y^2}{9} = 99.\\]",
    "The fourth term of a geometric series is $24$ and the eleventh term is $3072$. What is the common ratio?"
]
combined_prompts = [hierarchial_prompt + '\n' + prompt for prompt in prompts]
sampling_params = SamplingParams(temperature=0, top_p=1,max_tokens=1024)
llm = LLM(model=model_name, trust_remote_code=True)
outputs = llm.generate(combined_prompts, sampling_params)

#Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")
```

Here we also provide inference code with [vLLM](https://github.com/vllm-project/vllm) . vLLM is a fast and easy-to-use library for LLM inference and serving.

## Performance

We evaluate our SupperCorrect-7B on two widely used English math benchmarks GSM8K and MATH. All evaluations are tested with our evaluation method which is zero-shot hierarchical thought based prompting.

![image](imgs/table.png)

## Evaluation  

Here we provide two different evaluation methods: **online version**  which utilizes GPT-4o to conduct a more fair and robust judgement and **offline version**  which utilizes programming method to verify the final results. Both methods aim to provide a more accurate and strict evaluation results, as the final results in MATH dataset are not always numeric or pure expression. We now provide online version for evaluation, we will update soon for offline version.


```bash
API_KEY= "Input your key here"
MODEL_NAME_OR_PATH="BitStarWalkin/SuperCorrect-7B"
export CUDA_VISIBLE_DEVICES="0"
bash evaluation.sh $API_KEY $MODEL_NAME_OR_PATH

```

## Citation

```bash
@article{yang2024super,
title={SuperCorrect: Supervising and Correcting Language Models with Error-Driven Insights}
  author={},
  journal={arXiv preprint arXiv:2410.},
  year={2024}
}
@article{yang2024buffer,
  title={Buffer of Thoughts: Thought-Augmented Reasoning with Large Language Models},
  author={Yang, Ling and Yu, Zhaochen and Zhang, Tianjun and Cao, Shiyi and Xu, Minkai and Zhang, Wentao and Gonzalez, Joseph E and Cui, Bin},
  journal={arXiv preprint arXiv:2406.04271},
  year={2024}
}
```

## Acknowledgements

Our SuperCorrect is a two-stage fine-tuning model which based on several extraordinary open-source models like [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math), [DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math), [Llama3-Series](https://github.com/meta-llama/llama3). Our evaluation method is based on the code base of outstanding works like [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) and  [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). We also want to express our gratitude for amazing works such as [BoT](https://github.com/YangLing0818/buffer-of-thought-llm) which provides the idea of thought template.

