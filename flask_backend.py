from flask import Flask,jsonify,request # Add render_template
import xmlrpc.client
import json
from flask_cors import CORS
import logging
import requests

import sys
# sys.path.append('/home/milan/llm_server')
import grpc
from google.protobuf.json_format import MessageToDict
import llm_server_pb2 as sc
import llm_server_pb2_grpc as sc_grpc
import traceback

MAX_MESSAGE_LENGTH = 1024 * 1024 * 8

channel = grpc.insecure_channel('localhost:8018', options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ])
llm_server = sc_grpc.LLMServiceStub(channel)

deadline_seconds = 30

app = Flask(__name__)
CORS(app,supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'


server_t5 = xmlrpc.client.ServerProxy("http://localhost:8025")

backednURL = "http://216.151.16.6:5000/api/question"


handler = logging.FileHandler(filename='current_logs.log', encoding='utf-8')

FORMAT = '[%(levelname)s] %(asctime)s:%(name)s:  %(message)s'

logging.basicConfig(format=FORMAT, level=logging.INFO,handlers=[handler])

@app.route('/ask', methods=['POST'])
def call_full_asker_grpc():
    quest = request.json['quest']
    ip_addr = request.remote_addr
    print(quest)
    body = {"question":quest,"version":"v2"}
    headers = {"Content-Type": "application/json"}
    # answers = json.loads(answers)

    try:
        response = requests.post(backednURL,json=body,headers=headers)
        answers = response.json()
        print(type(answers))
        print(answers[0])

        answers_list = [sc.Answer(answer=answer['answer'],sentence=answer['sentence'], link=answer['link']) for answer in answers]

        res = llm_server.generate(sc.LLM_Input(question=quest,answers_list=answers_list),timeout=deadline_seconds)
        gen_text = res.generated_text
        dirty_text = res.dirty_text
        print(type(res.all_links))
        link_list = [MessageToDict(l) for l in res.all_links]
        full_req_time = res.end_time
        gen_time = res.end_gen_time
        question_type = res.question_type

    except grpc.RpcError as e:
        print(traceback.format_exc())
        logging.error(e)
        gen_text = "It takes me too long to answer, ask me again later. :("
        dirty_text = "Empty"
        link_list = []
        answers = []
        full_req_time = deadline_seconds
        gen_time = 100
        question_type = "Unknown"
    except Exception as aa:
        print(traceback.format_exc())
        logging.error(aa)
        gen_text = "Hmm... don\'t know"
        dirty_text = "Empty"
        link_list = []
        answers = []
        full_req_time = 100
        gen_time = 100
        question_type = "Unknown"

    toReturn = {'question':quest,'generation':gen_text,'dirty_text':dirty_text,'link_list':link_list,'gen_time':gen_time,'backend_time':full_req_time,'question_type':question_type}
    # toReturn['question'] = quest
    toReturn['answers'] = answers
    toReturn['ip'] = ip_addr
    print(toReturn)
    response = jsonify(toReturn)



    print(toReturn)
    logging.info(json.dumps(toReturn))

    response.headers.add('Set-Cookie', 'HttpOnly;Secure;SameSite=None')

    return response

@app.route('/log', methods=['POST'])
def log_link_click():
    jj = request.json
    jj['user_agent'] = request.headers.get('User-Agent')
    logging.info(jj)
    return ('',200)

if __name__=="__main__":
    app.run(host='localhost', port=9009,debug=False)
