"""
Script to create sentences using the DeepSeek API from the dataset's tabular features.
"""
import pandas as pd
import numpy as np
import os
import random
import json

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# Set a global seed
set_seed(42)
from tqdm import tqdm
import shutup; shutup.please()


dataset = "Tramadol-related_mortalities"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
dat_dir = os.path.join(project_root, "dat")

df = pd.read_csv(os.path.join(dat_dir, dataset, "proc", "df_together.csv"), index_col=0)
df.drop(["Temp_sentence",'label'], axis=1, inplace=True)

df_new = pd.DataFrame({})

prompts_sentences = []

for i, row in df.iterrows():
    prompt = f"""
Your role is to convert this encoded adverse drug effect report into a single concise, well constructed sentence that will help discover causal effects!\nPlease inlcude every information and drug name from the report into the sentence.


Age: {row['age']}
Gender: {row["gender"]}
Primary suspect drug(s): {row["psd"]}
Dose: {row["dose"]}
Indication: {row["indication"]}
Adverse Drug Event: {row["ade"]}
Seondary suspect drug(s): {row["ssd"]}
Concomitant drug(s): {row["ccd"]}
Interacting drug(s): {row["idrug"]}"""
    #print(prompt)
    prompts_sentences.append(prompt)
    
df_new["sentence"] = prompts_sentences
df_new.index = df.index

api_key = "YOUR_DEEPSEEK_API_KEY_HERE"

from openai import OpenAI
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

llm_sentences = {}

for i in tqdm(range(len(df_new))):
    try:
        messages = [{"role": "user", "content": df_new["sentence"][i]}]
        response=client.chat.completions.create(model="deepseek-chat",messages=messages)
        llm_sentences[df_new.index[i]] = response.choices[0].message.content
    except:
        llm_sentences[df_new.index[i]] = "failed"
    
    if i % 100 == 0:
        with open(os.path.join(dat_dir, dataset, "proc", "llm_sentences.json"), "w") as f:
            json.dump(llm_sentences, f)

df_new["llm_sentence"] = df_new.index.map(llm_sentences)
df_new.to_csv(os.path.join(dat_dir, dataset, "proc", "df_together_with_llm_sentence.csv"))
    
