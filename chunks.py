# """def send_prompt(java_files, json_file, query):
#     Generates code based on a set of Java files, JSON configuration data, and a prompt or query.

#     Args:
#         java_files: A list of paths to Java files.
#         json_file: The path to a JSON file containing configuration data.
#         query: The prompt or query to be used for code generation.

#     Returns:
#         A string containing the generated code.


#     # Create a list to store the prompt chunks
#     prompt_chunks = []

#     # Iterate over the Java files and create prompt chunks
#     for file in java_files:
#         with open(file, 'r') as f:
#             java_code = f.read()
#             chunk = prompt_template.format(java_code=java_code, json_data=json.dumps(json_data, indent=4))
#             prompt_chunks.append(chunk)

#     # Add the query or prompt to the last chunk
#     prompt_chunks[-1] += "\n\n" + query

#     # Send each chunk to the model and concatenate the responses
#     responses = []
#     for chunk in prompt_chunks:
#         inputs = tokenizer.encode_plus(
#             chunk,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
#         outputs = model.generate(
#             inputs['input_ids'],
#             attention_mask=inputs['attention_mask'],
#             max_length=512
#         )
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         responses.append(response)

#     # Concatenate the responses into a single string
#     response = "\n".join(responses)

#     return response

# # Prompt for input
# java_files = input("Enter the Java files (comma-separated): ").split(",")
# json_file = input("Enter the JSON file: ")
# query = input("Enter the query: ")

# # Send the prompt and print the response
# response = send_prompt(java_files, json_file, query)
# print(response)
# """


# -----------------------------------------------------------------------
# # from transformers import AutoTokenizer

# from transformers import AutoTokenizer,BertModel
# model = BertModel.from_pretrained("bert-base-uncased")

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# # model = AutoModelForCausalLM.from_pretrained('gpt-3-5')

# def send_prompt(java_files, query):
#     """
#     Generates code based on a set of Java files and a prompt or query.

#     Args:
#         java_files: A list of paths to Java files.
#         query: The prompt or query to be used for code generation.

#     Returns:
#         A string containing the generated code.
#     """

#     prompt_template = """
# """

#     # Create a list to store the prompt chunks
#     prompt_chunks = []

#     # Iterate over the Java files and create prompt chunks
#     for file in java_files:
#         with open(file, 'r') as f:
#             java_code = f.read()
#             chunk = prompt_template.format(java_code=java_code)
#             prompt_chunks.append(chunk)

#     # Add the query or prompt to the last chunk
#     prompt_chunks[-1] += "\n\n" + query

#     # Send each chunk to the model and concatenate the responses
#     responses = []
#     for chunk in prompt_chunks:
#         inputs = tokenizer.encode_plus(
#             chunk,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
#         outputs = tokenizer.generate(
#             inputs['input_ids'],
#             attention_mask=inputs['attention_mask'],
#             max_length=512
#         )
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         responses.append(response)

#     # Concatenate the responses into a single string
#     response = "\n".join(responses)

#     return response

# # Prompt for input
# java_files = input("Enter the Java files (comma-separated): ").split(",")
# query = input("Enter the query: ")

# # Send the prompt and print the response
# response = send_prompt(java_files, query)
# print(response)


# -----------------------------------------------------------------------

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Load the CodeLlama-34-b-Instruct model and tokenizer
# model = AutoModelForCausalLM.from_pretrained("CodeLlama-34-b-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("CodeLlama-34-b-Instruct")

# def send_prompt(java_files, query):
#     """
#     Generates code based on a set of Java files and a prompt or query.

#     Args:
#         java_files: A list of paths to Java files.
#         query: The prompt or query to be used for code generation.

#     Returns:
#         A string containing the generated code.
#     """

#     prompt_template = ""

#     # Create a list to store the prompt chunks
#     prompt_chunks = []

#     # Iterate over the Java files and create prompt chunks
#     for file in java_files:
#         with open(file, 'r') as f:
#             java_code = f.read()
#             chunk = prompt_template.format(java_code=java_code)
#             prompt_chunks.append(chunk)

#     # Add the query or prompt to the last chunk
#     prompt_chunks[-1] += "\n\n" + query

#     # Send each chunk to the model and concatenate the responses
#     responses = []
#     for chunk in prompt_chunks:
#         inputs = tokenizer.encode_plus(
#             chunk,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
#         outputs = model.generate(
#             inputs['input_ids'],
#             attention_mask=inputs['attention_mask'],
#             max_length=512
#         )
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         responses.append(response)

#     # Concatenate the responses into a single string
#     response = "\n".join(responses)

#     return response

# # Prompt for input
# java_files = input("Enter the Java files (comma-separated): ").split(",")
# query = input("Enter the query: ")

# # Send the prompt and print the response
# response = send_prompt(java_files, query)
# print(response)

# -----------------------------------------------------------------------

# import sys
# sys.path.insert(0, '/home/dockuser1/kriti/llama/')
# import generation
# import generation.tokenizer as tokenizer
# import generation.model as model
# import generation._init_ as init

# from generation.model import ModelArgs, Transformer
# from generation.tokenizer import Tokenizer

# def send_prompt(java_files: List[str], query: str, model_path: str, tokenizer_path: str, ckpt_dir: str):
#     """
#     Generates code based on a set of Java files and a prompt or query.

#     Args:
#         java_files: A list of paths to Java files.
#         query: The prompt or query to be used for code generation.
#         model_path: The path to the model file.
#         tokenizer_path: The path to the tokenizer file.
#         ckpt_dir: The directory containing the model checkpoint.

#     Returns:
#         A string containing the generated code.
#     """

#     # Initialize the model and tokenizer
#     model_args = model.ModelArgs()
#     model_args.vocab_size = 50000  # adjust this value as needed
#     model_ = model.Transformer(model_args)
#     tokenizer_ = tokenizer.Tokenizer(tokenizer_path)

#     # Create a Llama instance
#     llama = model.Llama(model_, tokenizer_)

#     # Load the model checkpoint
#     llama = model.Llama.build(ckpt_dir, tokenizer_path, model_args.max_seq_len, model_args.max_batch_size)

#     # Create a list to store the prompt chunks
#     prompt_chunks = []

#     # Iterate over the Java files and create prompt chunks
#     for file in java_files:
#         with open(file, 'r') as f:
#             java_code = f.read()
#             chunk = java_code
#             prompt_chunks.append(chunk)

#     # Add the query or prompt to the last chunk
#     prompt_chunks[-1] += "\n\n" + query

#     # Send each chunk to the model and concatenate the responses
#     responses = []
#     for chunk in prompt_chunks:
#         generation_tokens = llama.text_completion([chunk], temperature=0.6, top_p=0.9, max_gen_len=512)
#         response = llama.tokenizer.decode_infilling(generation_tokens[0]['generation']['tokens'])
#         responses.append(response)

#     # Concatenate the responses into a single string
#     response = "\n".join(responses)

#     return response

# # Prompt for input
# java_files = input("Enter the Java files (comma-separated): ").split(",")
# query = input("Enter the query: ")

# # Send the prompt and print the response
# response = send_prompt(java_files, query, "/path/to/model/model.pt", "/path/to/tokenizer/tokenizer.model", "/path/to/model/checkpoint")
# print(response)
# ------------------------------------------------------------------------

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# def send_prompt(java_files, query, prompt_template, model_name='t5-base'):
#     # Load the model and tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     # Create a list to store the prompt chunks
#     prompt_chunks = []

#     # Iterate over the Java files and create prompt chunks
#     for file in java_files:
#         with open(file, 'r') as f:
#             java_code = f.read()
#             chunk = prompt_template.format(java_code=java_code)
#             prompt_chunks.append(chunk)

#     # Add the query or prompt to the last chunk
#     prompt_chunks[-1] += "\n\n" + query

#     # Send each chunk to the model and concatenate the responses
#     responses = []
#     for chunk in prompt_chunks:
#         inputs = tokenizer.encode_plus(
#             chunk,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
#         outputs = model.generate(
#             inputs['input_ids'],
#             attention_mask=inputs['attention_mask'],
#             max_length=512
#         )
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         responses.append(response)

#     # Concatenate the responses into a single string
#     response = "\n".join(responses)

#     return response

# # Example usage
# # Take file paths as input from the user
# java_files_input = input("Enter the paths of the Java files, separated by commas: ")
# java_files = [file.strip() for file in java_files_input.split(',')]
# query = input("Enter your query: ")
# prompt_template = """
# Java Code:
# {java_code}
# """

# response = send_prompt(java_files, query, prompt_template)
# print("output/response of prompt:",end ="\n")
# print(response)


# ------------------------------------------------------------------------



# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import concat_prompt
import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_seq_len: int = 256,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    prompt_str = concat_prompt.prompt_str()
    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
       f""" + {prompt_str} + """
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)



