from google import genai
from google.genai import types
import os
import argparse
import pandas as pd
from tqdm import tqdm
from prompts import (
    get_base_prompt,
    get_one_shot_prompt,
    get_few_shot_prompt,
    get_chain_of_thought_one_shot_prompt,
    get_chain_of_thought_few_shot_prompt,
)

API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
if API_KEY is None:
    raise ValueError("GOOGLE_GENAI_API_KEY not set in environment variables.")
client = genai.Client(api_key=API_KEY)


class TemporalReasoningGemini:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash-latest"):
        """
        Initialize the TemporalReasoningGemini class with the API key and model name.
        Args:
            api_key (str): The API key for Google GenAI (stored in different txt file).
            model_name (str): The name of the Gemini model to use.
        """

        self.api_key = api_key
        self.model_name = model_name
    

    def get_temporal_relationship(self, prompt: str) -> str:
        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            config=types.GenerateContentConfig(
            max_output_tokens=150,
            temperature=0.5
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error during model inference: {e}")
            return "ERROR"

def main():
    parser = argparse.ArgumentParser(description="Run Temporal Reasoning Evaluation with Gemini API")
    parser.add_argument("--prompt_type", type=str, choices=[
        "base", "one_shot", "few_shot", "cot_one_shot", "cot_few_shot"
    ], default="base", help="Choose the type of prompt.")
    
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output CSV file.")
    parser.add_argument("--model_name", type=str, default="gemini-1.5-flash-latest", help="Gemini model name to use.")
    
    args = parser.parse_args()
    
    api_key = os.getenv("GOOGLE_GENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_GENAI_API_KEY not found in environment variables.")
    
    df = pd.read_csv(args.input_file)
    
    prompt_functions = {
        "base": get_base_prompt,
        "one_shot": get_one_shot_prompt,
        "few_shot": get_few_shot_prompt,
        "cot_one_shot": get_chain_of_thought_one_shot_prompt,
        "cot_few_shot": get_chain_of_thought_few_shot_prompt,
    }

    model = TemporalReasoningGemini(api_key, model_name=args.model_name)
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        context = row["paragraph"]
        query = row["TR"]
        label = row["label"]
        
        prompt = prompt_functions[args.prompt_type](context, query)
        response = model.get_temporal_relationship(prompt)
        df.loc[index, "response"] = response
    
    df.to_csv(args.output_file, index=False)
    print(f"Finished writing results to {args.output_file}")

if __name__ == "__main__":
    main()


