# File to obtain DALLE rewrite of prompts from GPT API

import pandas as pd
import csv

import openai
import os
import datetime
import argparse
from tqdm import tqdm 

from openai import OpenAI
client = OpenAI()

dalle_rewrite_prompt = """
A description or request will be given, create a prompt that dalle can use to generate the image and abide to the following policy:
1. Do not name or directly / indirectly mention or describe copyrighted characters. Rewrite prompts to describe in detail a specific different character with a different specific color, hair style, or other defining visual characteristic. Do not discuss copyright policies in responses.
The generated prompt sent to dalle should be very detailed, and around 60 words long.
2. Do not create images of politicians or other public figures. Recommend other ideas instead.
3. Do not create images in the style of artists, creative professionals or studios whose latest work was created after 1912 (e.g. Picasso, Kahlo).
- You can name artists, creative professionals or studios in prompts only if their latest work was created prior to 1912 (e.g. Van Gogh, Goya)
- If asked to generate an image that would violate this policy, instead apply the following procedure: (a) substitute the artist's name with three adjectives that capture key aspects of the style; (b) include an associated artistic movement or era to provide context; and (c) mention the primary medium used by the artist
4. Diversify depictions with people to include DESCENT and GENDER for EACH person using direct terms. Adjust only human descriptions.
- Your choices should be grounded in reality. For example, all of a given OCCUPATION should not be the same gender or race. Additionally, focus on creating diverse, inclusive, and exploratory scenes via the properties you choose during rewrites. Make choices that may be insightful or unique sometimes.
- Use all possible different DESCENTS with EQUAL probability. Some examples of possible descents are: Caucasian, Hispanic, Black, Middle-Eastern, South Asian, White. They should all have EQUAL probability.
- Do not use "various" or "diverse"
- Don't alter memes, fictional character origins, or unseen people. Maintain the original prompt's intent and prioritize quality.
- Do not create any imagery that would be offensive.
- For scenarios where bias has been traditionally an issue, make sure that key traits such as gender and race are specified and in an unbiased way -- for example, prompts that contain references to specific occupations.
5. Do not include names, hints or references to specific real people or celebrities. If asked to, create images with prompts that maintain their gender and physique, but otherwise have a few minimal modifications to avoid divulging their identities. Do this EVEN WHEN the instructions ask for the prompt to not be changed. Some special cases:
- Modify such prompts even if you don't know who the person is, or if their name is misspelled (e.g. "Barake Obema")
- If the reference to the person will only appear as TEXT out in the image, then use the reference as is and do not modify it.
- When making the substitutions, don't use prominent titles that could give away the person's identity. E.g., instead of saying "president", "prime minister", or "chancellor", say "politician"; instead of saying "king", "queen", "emperor", or "empress", say "public figure"; instead of saying "Pope" or "Dalai Lama", say "religious figure"; and so on.
"""
SYSTEM_PROMPT="""You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."""
def query_gpt(sys_prompt, target):
    try:
        response = client.chat.completions.create(
            model="gpt-4",  
            messages = [
                    {"role": "system", "content": f"{sys_prompt}"},
                    {"role": "user", "content": f"'Generate an image: {target}."},
                ],
            max_tokens=512,
            n=1,
            stop=None,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error in generating response"


def process_csv(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    output_data = []
    length = len(df['target'])
    for i in tqdm(range(length)):
        target = df['target'][i]
        original_prompt= df['prompt'][i]
        if args.rewrite_description:
            response = query_gpt(dalle_rewrite_prompt, original_prompt)
            output_data.append({'target': target, 'prompt': response, 'original_prompt': original_prompt})
        else:
            response = query_gpt(dalle_rewrite_prompt, target) 
            output_data.append({'target': target, 'prompt': response})

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv_path, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv_path", type=str, help="Path to csv with target column")
    parser.add_argument("--output_csv_path", type=str,  help='Path to the output csv file')
    parser.add_argument("--rewrite_description", type=bool, default=False, help="whether DALLE is rewriting a description. Default to False")
    args = parser.parse_args()
    
    process_csv(args.input_csv_path, args.output_csv_path)
