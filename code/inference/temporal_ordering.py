"""
This script is used to run language models (HF) on a given dataset to predict the temporal relationship between two events.
The script uses the Hugging Face Transformers library to run the model and generate predictions.
The script takes the following arguments: prompt_type, model_name, cache_dir, access_token, input_file, output_file.
"""


import os
import transformers
import torch
import argparse
import pandas as pd
#from inference.config import model_name, cache_dir, access_token, input_file
from prompts import (
    get_base_prompt,
    get_one_shot_prompt,
    get_few_shot_prompt,
    get_chain_of_thought_one_shot_prompt,
    get_chain_of_thought_few_shot_prompt,
)

class TemporalReasoningModel:
    def __init__(self, model_name, cache_dir, access_token):
        os.environ['HF_HOME'] = f"{cache_dir}/huggingface/"
        self.model_name = model_name
        self.access_token = access_token
        self.pipeline = transformers.pipeline(
            "text-generation", 
            model=self.model_name, 
            model_kwargs={"torch_dtype": torch.bfloat16}, 
            #device="cuda", 
            token=self.access_token
        )
    
    def get_temporal_relationship(self, prompt):
        outputs = self.pipeline(prompt, max_new_tokens=256)
        print(outputs)
        return outputs[0]["generated_text"][-1]["content"]


def main():
    print("Script started")

    ## Arguments
    # run --help to read the description of each argument, and possible values. 
    parser = argparse.ArgumentParser(description="Run Temporal Reasoning Model with different prompt types. \
                                     Arguments: prompt_type, model_name, cache_dir, access_token, input_file, output_file")
    
    parser.add_argument("--prompt_type", type=str, choices=[
        "base", "one_shot", "few_shot", "cot_one_shot", "cot_few_shot"
    ], default="base", help="Choose the type of prompt to use between: base, one_shot, few_shot, cot_one_shot, cot_few_shot")

    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="HF Model name to use for inference.")

    parser.add_argument("--cache_dir", type=str, default="/.cache", help="Cache directory to store HF models (to change depending on the cluster filesystem).")

    parser.add_argument("--access_token", type=str, help="Hugging Face API access token.")

    parser.add_argument("--input_file", type=str, help="Input CSV file containing the data to run inference on.")

    parser.add_argument("--output_file", type=str, help="Output CSV file to store the results of the inference.")
    
    args = parser.parse_args()
    model_name = args.model_name
    cache_dir = args.cache_dir
    access_token = args.access_token
    input_file = args.input_file
    output_file = args.output_file



    # Load data from CSV
    # get input file path as argument when running this script
    df = pd.read_csv(input_file)
    
    # Select the appropriate prompt template
    prompt_functions = {
        "base": get_base_prompt,
        "one_shot": get_one_shot_prompt,
        "few_shot": get_few_shot_prompt,
        "cot_one_shot": get_chain_of_thought_one_shot_prompt,
        "cot_few_shot": get_chain_of_thought_few_shot_prompt,
    }

    model = TemporalReasoningModel(model_name, cache_dir, access_token)
    
    for index, row in df.iterrows():
        
        context = row["paragraph"]
        query = row["TR"]
        label = row["label"]
        #print("-----Context: ", context)
        #print("-----Query: ", query)
        #print("-----Label: ", label)
        
        selected_prompt = prompt_functions[args.prompt_type](context, query)
        response = model.get_temporal_relationship(selected_prompt)
        #print(f"Row {index}: Model Response ({args.prompt_type}) - {response}, Label: {label}")
        # write the response to the output file
        df.loc[index, "response"] = response
        df.to_csv(output_file, index=False)



if __name__ == "__main__":
    main()
