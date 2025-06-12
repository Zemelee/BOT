import os
import copy
from functools import lru_cache
import json
import aioboto3
import aiohttp
import numpy as np
import ollama

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    Timeout,
    AsyncAzureOpenAI,
)

import base64
import struct

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from pydantic import BaseModel, Field
from typing import List, Dict, Callable, Any
from .base import BaseKVStorage
from .utils import compute_args_hash, wrap_embedding_func_with_attrs

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=30), # 增加重试间隔时间
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)

# 如果已经缓存过这次请求的结果，就直接返回缓存结果；否则调用 LLM 并缓存。
async def openai_complete_if_cache(
    model,
    prompt, # '识别查询内容中的高/低层关键词'+'蒸馏提取到的任务关键信息'
    system_prompt=None,
    history_messages=[],
    base_url='https://api.moonshot.cn/v1',
    api_key='sk-',
    **kwargs,
) -> str:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    # 如果缓存中存在这个 key，就直接返回缓存结果，不再调用模型
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    openai_async_client = AsyncOpenAI(base_url=base_url)
    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    # high: ["Problem analysis", "Thought templates", "Task description", "Solution generation"] 
    # # low: ["Raymond", "Samantha", "Age difference", "Son\'s birth year", "Current year", "Python transformation", "Input parameters"]
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
    return response.choices[0].message.content



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)



class BedrockError(Exception):
    """Generic error for issues related to Amazon Bedrock"""


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, max=60),
    retry=retry_if_exception_type((BedrockError)),
)
async def bedrock_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    aws_access_key_id=None,
    aws_secret_access_key=None,
    aws_session_token=None,
    **kwargs,
) -> str:
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get(
        "AWS_ACCESS_KEY_ID", aws_access_key_id
    )
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get(
        "AWS_SECRET_ACCESS_KEY", aws_secret_access_key
    )
    os.environ["AWS_SESSION_TOKEN"] = os.environ.get(
        "AWS_SESSION_TOKEN", aws_session_token
    )

    # Fix message history format
    messages = []
    for history_message in history_messages:
        message = copy.copy(history_message)
        message["content"] = [{"text": message["content"]}]
        messages.append(message)

    # Add user prompt
    messages.append({"role": "user", "content": [{"text": prompt}]})

    # Initialize Converse API arguments
    args = {"modelId": model, "messages": messages}

    # Define system prompt
    if system_prompt:
        args["system"] = [{"text": system_prompt}]

    # Map and set up inference parameters
    inference_params_map = {
        "max_tokens": "maxTokens",
        "top_p": "topP",
        "stop_sequences": "stopSequences",
    }
    if inference_params := list(
        set(kwargs) & set(["max_tokens", "temperature", "top_p", "stop_sequences"])
    ):
        args["inferenceConfig"] = {}
        for param in inference_params:
            args["inferenceConfig"][inference_params_map.get(param, param)] = (
                kwargs.pop(param)
            )

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    # Call model via Converse API
    session = aioboto3.Session()
    async with session.client("bedrock-runtime") as bedrock_async_client:
        try:
            response = await bedrock_async_client.converse(**args, **kwargs)
        except Exception as e:
            raise BedrockError(e)

        if hashing_kv is not None:
            await hashing_kv.upsert(
                {
                    args_hash: {
                        "return": response["output"]["message"]["content"][0]["text"],
                        "model": model,
                    }
                }
            )

        return response["output"]["message"]["content"][0]["text"]


@lru_cache(maxsize=1)
def initialize_hf_model(model_name):
    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    return hf_model, hf_tokenizer


async def hf_model_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model_name = model
    hf_model, hf_tokenizer = initialize_hf_model(model_name)
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    input_prompt = ""
    try:
        input_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        try:
            ori_message = copy.deepcopy(messages)
            if messages[0]["role"] == "system":
                messages[1]["content"] = (
                    "<system>"
                    + messages[0]["content"]
                    + "</system>\n"
                    + messages[1]["content"]
                )
                messages = messages[1:]
                input_prompt = hf_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            len_message = len(ori_message)
            for msgid in range(len_message):
                input_prompt = (
                    input_prompt
                    + "<"
                    + ori_message[msgid]["role"]
                    + ">"
                    + ori_message[msgid]["content"]
                    + "</"
                    + ori_message[msgid]["role"]
                    + ">\n"
                )

    input_ids = hf_tokenizer(
        input_prompt, return_tensors="pt", padding=True, truncation=True
    ).to("cuda")
    inputs = {k: v.to(hf_model.device) for k, v in input_ids.items()}
    output = hf_model.generate(
        **input_ids, max_new_tokens=512, num_return_sequences=1, early_stopping=True
    )
    response_text = hf_tokenizer.decode(
        output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    )
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": response_text, "model": model}})
    return response_text


async def ollama_model_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)

    ollama_client = ollama.AsyncClient()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await ollama_client.chat(model=model, messages=messages, **kwargs)

    result = response["message"]["content"]

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": model}})

    return result


@lru_cache(maxsize=1)
def initialize_lmdeploy_pipeline(
    model,
    tp=1,
    chat_template=None,
    log_level="WARNING",
    model_format="hf",
    quant_policy=0,
):
    from lmdeploy import pipeline, ChatTemplateConfig, TurbomindEngineConfig

    lmdeploy_pipe = pipeline(
        model_path=model,
        backend_config=TurbomindEngineConfig(
            tp=tp, model_format=model_format, quant_policy=quant_policy
        ),
        chat_template_config=ChatTemplateConfig(model_name=chat_template)
        if chat_template
        else None,
        log_level="WARNING",
    )
    return lmdeploy_pipe


async def lmdeploy_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    chat_template=None,
    model_format="hf",
    quant_policy=0,
    **kwargs,
) -> str:
    """
    Args:
        model (str): The path to the model.
            It could be one of the following options:
                    - i) A local directory path of a turbomind model which is
                        converted by `lmdeploy convert` command or download
                        from ii) and iii).
                    - ii) The model_id of a lmdeploy-quantized model hosted
                        inside a model repo on huggingface.co, such as
                        "InternLM/internlm-chat-20b-4bit",
                        "lmdeploy/llama2-chat-70b-4bit", etc.
                    - iii) The model_id of a model hosted inside a model repo
                        on huggingface.co, such as "internlm/internlm-chat-7b",
                        "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                        and so on.
        chat_template (str): needed when model is a pytorch model on
            huggingface.co, such as "internlm-chat-7b",
            "Qwen-7B-Chat ", "Baichuan2-7B-Chat" and so on,
            and when the model name of local path did not match the original model name in HF.
        tp (int): tensor parallel
        prompt (Union[str, List[str]]): input texts to be completed.
        do_preprocess (bool): whether pre-process the messages. Default to
            True, which means chat_template will be applied.
        skip_special_tokens (bool): Whether or not to remove special tokens
            in the decoding. Default to be True.
        do_sample (bool): Whether or not to use sampling, use greedy decoding otherwise.
            Default to be False, which means greedy decoding will be applied.
    """
    try:
        import lmdeploy
        from lmdeploy import version_info, GenerationConfig
    except Exception:
        raise ImportError("Please install lmdeploy before intialize lmdeploy backend.")

    kwargs.pop("response_format", None)
    max_new_tokens = kwargs.pop("max_tokens", 512)
    tp = kwargs.pop("tp", 1)
    skip_special_tokens = kwargs.pop("skip_special_tokens", True)
    do_preprocess = kwargs.pop("do_preprocess", True)
    do_sample = kwargs.pop("do_sample", False)
    gen_params = kwargs

    version = version_info
    if do_sample is not None and version < (0, 6, 0):
        raise RuntimeError(
            "`do_sample` parameter is not supported by lmdeploy until "
            f"v0.6.0, but currently using lmdeloy {lmdeploy.__version__}"
        )
    else:
        do_sample = True
        gen_params.update(do_sample=do_sample)

    lmdeploy_pipe = initialize_lmdeploy_pipeline(
        model=model,
        tp=tp,
        chat_template=chat_template,
        model_format=model_format,
        quant_policy=quant_policy,
        log_level="WARNING",
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    gen_config = GenerationConfig(
        skip_special_tokens=skip_special_tokens,
        max_new_tokens=max_new_tokens,
        **gen_params,
    )

    response = ""
    async for res in lmdeploy_pipe.generate(
        messages,
        gen_config=gen_config,
        do_preprocess=do_preprocess,
        stream_response=False,
        session_id=1,
    ):
        response += res.response

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": response, "model": model}})
    return response



async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "moonshot-v1-32k",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def bedrock_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await bedrock_complete_if_cache(
        "anthropic.claude-3-haiku-20240307-v1:0",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def hf_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await hf_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def ollama_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await ollama_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)

async def custom_embedding(texts: list[str], model_name: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Handle different model output types
        if hasattr(outputs, 'last_hidden_state'):
            embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            embedding = outputs[0].mean(dim=1).numpy()
        else:
            raise ValueError("Unsupported model output type for embedding extraction")
        embeddings.append(embedding[0])
    return np.array(embeddings)

@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)


async def hf_embedding(texts: list[str], tokenizer, embed_model) -> np.ndarray:
    input_ids = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).input_ids
    with torch.no_grad():
        outputs = embed_model(input_ids)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()


