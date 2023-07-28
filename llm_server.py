from time import time
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
import traceback

import multiprocessing
from concurrent import futures
import argparse

import os
import sys
import grpc
import llm_server_pb2 as sc
import llm_server_pb2_grpc as sc_grpc
import re

NUM_WORKERS = 1


MAX_MESSAGE_LENGTH = 1024 * 1024 * 8
pattern = r"\[SENT:(\d+)\]"
device = torch.device("cpu")
# device = "cuda:0"
print(device)

access_token = "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"

def remove_prefix_strings(lst):
    """
    Remove all strings that are prefixes of some other string in the list
    """
    for s1 in lst[:]:
        for s2 in lst:
            if s1 != s2 and s1 in s2:
                lst.remove(s1)
                break
    return lst

def create_text_with_list(tokenizer,model,answers_list,quest,use_gramformer=False):
    start_time = time()
    sentences = [a['sentence'].strip().strip('.').strip() for a in answers_list]
    # sentences = sentences[0:1] + sentences[0:1] + sentences[0:1] + sentences[1:2] + sentences[1:2] + sentences[2:3] + sentences[2:3] + sentences[3:]
    context = '. '.join(sentences)
    context = quest + '? ' + context + "."

    answers = []
    list_of_answers = []
    for ind,a in enumerate(answers_list):
        answer,sentence = a['answer'].strip().strip('.').strip(),a['sentence'].strip().strip('.').strip()
        start = sentence.find(answer)
        list_of_answers.append(answer)
        if start == -1:
            # print('error',answer,'  :::  ',sentence)
            continue
        end = start + len(answer)
        tmp_highlight = sentence[:start] + '**' + answer + '**' + sentence[end:] + f'[SENT:{ind}]'
        answers.append(tmp_highlight)

    context_highlight = '. '.join(answers[:10])
    context_highlight = quest + '? ' + context_highlight + "."
    
    start_gen_time = time()
    generated_text = crate_text(context_highlight,tokenizer,model)[0]
    end_gen_time = time()-start_gen_time
    dirty_text = generated_text
    print(dirty_text)

    all_links = []
    offset = 0
    matches = re.finditer(pattern, generated_text)
    # print(matches)
    for match in matches:
        sentence_number = int(match.group(1))
        if sentence_number < 0 or sentence_number > len(answers_list):
            print(sentence_number)
            continue
        link = answers_list[sentence_number]['link']
        sentence_start = match.start() - offset  # Find the start index of the match
        sentence_end = match.end() - offset  # Find the end index of the sentence
        
        all_links.append({'position':sentence_start,'link':link})
        generated_text = generated_text[:sentence_start] + generated_text[sentence_end:]  # Remove the pattern and adjust the offset
        offset += len(match.group(0))  # Adjust the

    last_dot = generated_text.rfind('.')

    generated_text = generated_text[:last_dot]
    
    if not generated_text.strip().endswith('.'):
        generated_text = generated_text + '.'

    end_time = time() - start_time
    return generated_text,dirty_text,all_links,end_time,end_gen_time



def crate_text(text,tokenizer,model):
    prefix = "summarize: "

    text = prefix + text.replace('/n',' ')
    text_toc = tokenizer(text, return_tensors="pt").input_ids
    text_toc = text_toc.to(device)
    print(len(text_toc))

    if len(text_toc)>0 and (len(text_toc))>1023:
        print(f"ERROR: {text_toc}")
    
    out_token = model.generate(text_toc, max_new_tokens=200,num_beams=3, do_sample=False)
    print(len(out_token[0]))
    for t in out_token:
        print(tokenizer.convert_ids_to_tokens(t),end='\t')
    gen_text = tokenizer.decode(out_token[0], skip_special_tokens=True)
    dirty_get_text = gen_text
    
    dot_position = gen_text.find('.')
    quest_position = gen_text.find('?')
    if quest_position != -1 and dot_position > quest_position:
        gen_text = gen_text[quest_position+1:]
    gen_text = [t.strip() for t in gen_text.split('.')]
    gen_text_dict = {t:len(gen_text) for t in gen_text}
    gen_text_dict = {t:min(ind,gen_text_dict[t]) for ind,t in enumerate(gen_text) if t in gen_text_dict}
    sorted_gen_text_list = sorted(gen_text_dict.items(), key=lambda item: item[1])
    sorted_gen_text_list = [t for t,ind in sorted_gen_text_list]
    full_get_text = '. '.join(sorted_gen_text_list)


    sorted_gen_text_list = remove_prefix_strings(sorted_gen_text_list)
    gen_text = '. '.join(sorted_gen_text_list)

    return dirty_get_text,full_get_text,gen_text

class LLM_Server(sc_grpc.LLMServiceServicer):

    def __init__(self,dir_path):
        

        # dir_path = 'google/flan-t5-large_19k/final_model'
        self.model = AutoModelForSeq2SeqLM.from_pretrained("YottaAnswers/yotta-t5-large-three-tasks", torch_dtype=torch.bfloat16 if args.bf16 else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained("YottaAnswers/yotta-t5-large-three-tasks")

        self.model = self.model.to(device)
    
    def generate(self,request, context) -> str:
        print("request")
        question = request.question
        answer_list = [{'answer':a.answer,'sentence':a.sentence,'link':a.link,} for a in request.answers_list]
        # print(answer_list)
        use_gramformer = False
        if request.HasField('use_gramformer'):
            use_gramformer = request.use_gramformer
        try:
            generated_text,dirty_text,all_links,end_time,end_gen_time = create_text_with_list(self.tokenizer,self.model,answer_list,question,use_gramformer)
        except Exception as e:
            traceback.print_exc()
            return sc.LLM_Output()
        
        return sc.LLM_Output(generated_text=generated_text,dirty_text=dirty_text,all_links=all_links,end_time=end_time,end_gen_time=end_gen_time)


def _run_server(bind_address):
    # logger.debug(f"Server started. Awaiting jobs...")
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1),
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.so_reuseport", 0),
        ],
    )

    sc_grpc.add_LLMServiceServicer_to_server(
            LLM_Server('../google/flan-t5-large_19k/final_model'),
            server
        )
    
    server.add_insecure_port(bind_address)
    server.start()
    print(f'server started on {bind_address} with [PID {os.getpid()}]')
    server.wait_for_termination()

def serve(address = 'localhost', port_num = 8000):
    # with _reserve_port(port_num) as port:
        # sys.stdout.flush()
        workers = []

        bind_address = f"{address}:{port_num}"
        # _run_server(bind_address)
        for i in range(args.workers):

            ### use this bind_address for local port connection ###
            bind_address = f"{address}:{port_num+i}"
            
            ### use this bind address for unix socket connection ###
            # bind_address = f'unix:///var/run/yottaanswers/backend_{port_num+i}.sock'

            worker = multiprocessing.Process(target=_run_server, args=(bind_address,))
            worker.start()
            workers.append(worker)
        try:
            for worker in workers:
                worker.join()
        finally:
            for worker in workers:
                worker.terminate()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--port', type=int, default=8018)
    parser.add_argument('--address', type=str, default='localhost', choices=['localhost', '0.0.0.0'])
    parser.add_argument('--workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--bf16',action='store_true')

    args = parser.parse_args()

    if len(sys.argv) == 2 and sys.argv[1]=='test':
        # run_tests()
        serve(args.address, args.port)
    else:
        serve(args.address, args.port)
