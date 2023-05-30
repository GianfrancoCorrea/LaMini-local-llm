from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import pipeline
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSeq2SeqLM
import accelerate
import textwrap
import streamlit as st

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

# models
checkpoint = "MBZUAI/LaMini-Flan-T5-783M" 
# checkpoint = "MBZUAI/LaMini-T5-223M" 
# checkpoint = "MBZUAI/LaMini-Neo-1.3B" 
# checkpoint = "MBZUAI/LaMini-Neo-125m" 
# checkpoint = "MBZUAI/LaMini-GPT-1.5B" 

# device map is used to specify which layers should be loaded on which device (cpu or gpu)
# some models are too large to fit on a gpu, so we load them on cpu
device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": "cpu",
    "transformer.h": 0,
    "transformer.ln_f": 0,
    "transformer.wte": "cpu",
    "transformer.wpe": "cpu",
    "shared.weight": "cpu",
    "encoder.embed_tokens.weight": 0,
    "encoder": "cpu",
    "decoder": "cpu",
}

# automatic device map inference, use this if you don't know which layers should be loaded on which device
# config = AutoConfig.from_pretrained(checkpoint)
# with accelerate.init_empty_weights():
#     fake_model = AutoModelForSeq2SeqLM.from_config(config)
#     device_map = accelerate.infer_auto_device_map(fake_model)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# the base model is loaded with transformers, and then passed to the pipeline
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map=device_map,
    cache_dir='./cache', # cache dir is used to store the model weights during quantization
    quantization_config=quantization_config, # quantization config is used to specify the quantization parameters, such as the precision of the weights and activations
    torch_dtype=torch.float16, # torch_dtype is used to specify the precision of the weights
    load_in_8bit=True, # load_in_8bit is used to specify the precision of the activations
    # offload_folder="offload", # offload_folder is used to specify the folder where the quantized model will be stored
    )

pipe = pipeline('text2text-generation', 
    model = base_model,
    tokenizer = tokenizer,
    max_length=1024,
    do_sample=True,
    pad_token_id= 50256,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
    )


# prompt template
def get_prompt(instruction):
    prompt_template = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
    return prompt_template

# parse text
def parse_text(data):
    for item in data:
        text = item['generated_text']
        assistant_text_index = text.find('### Response:')
        if assistant_text_index != -1:
            assistant_text = text[assistant_text_index+len('### Response:'):].strip()
            wrapped_text = textwrap.fill(assistant_text, width=100)
            print(wrapped_text +'\n\n')
            st.write(wrapped_text +'\n\n')
            # return assistant_text

# prompt = 'What are the differences between alpacas, vicunas and llamas?'
# generated_text = pipe(get_prompt(prompt))
# parse_text(generated_text)

st.title(checkpoint)

# required prompt input
prompt = st.text_input('Plug in your prompt here')

if prompt:
    generated_text = pipe(get_prompt(prompt))
    st.write(generated_text)
    # parse_text(generated_text)