import json
from time import time
import requests
from tqdm import tqdm

backednURL = "http://localhost:9009/ask"

def call_api(quest):
    body = {"quest":quest}
    headers = {"Content-Type": "application/json"}
    start_time = time()
    response = requests.post(backednURL,json=body,headers=headers)
    end_time = time() - start_time
    response = response.json()
    gen_time = response['gen_time']
    return gen_time, end_time

with open("all_quest_filtered.txt",'r') as ff:
    lines = ff.readlines()

print(lines[0].strip())

# quest = "what is geng geng?"
lines = lines[:100]
start_time = time()

all_quest_len = len(lines)
print(all_quest_len)
maxt = 0.0
mint = 100.0

time_sum = 0.0
gen_time_sum = 0.0

for quest in tqdm(lines):
    quest = quest.strip()
    gin_time,in_time = call_api(quest)

    time_sum += in_time
    gen_time_sum += gin_time

    maxt = max(maxt,in_time)
    mint = min(mint,in_time)

end_time = time() - start_time

outp = {"full_time":end_time,"avg_time":time_sum/all_quest_len,"gen_avg_time":gen_time_sum/all_quest_len,"max_time":maxt,"min_time":mint}

with open("t5_large_on_gpu_stats.json","w") as ff:
    ff.write(json.dumps(outp))