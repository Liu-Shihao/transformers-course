from llama_index import ServiceContext
from llama_index.prompts.prompts import SimpleInputPrompt
import torch
from llama_index.llm_predictor import HuggingFaceLLMPredictor
"""
LlamaIndex 支持直接使用 HuggingFace 的 LLM

stabilityai/stablelm-tuned-alpha-3b
https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b/tree/main
"""

system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")


stablelm_predictor = HuggingFaceLLMPredictor(
    max_input_size=4096,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=False,
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b",
    model_name="StabilityAI/stablelm-tuned-alpha-3b",
    device_map="auto",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 4096},
    # uncomment this if using CUDA to reduce memory usage
    # model_kwargs={"torch_dtype": torch.float16}
)
service_context = ServiceContext.from_defaults(chunk_size_limit=1024, llm_predictor=stablelm_predictor)