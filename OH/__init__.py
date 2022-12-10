'''
pip install flask

'''

from flask import Flask,request,jsonify
import os
import shutil

# 챗봇 model 불러오기
from ChatBot.ChatBot import aichatbot

# STTeng model 불러오기
from Voice_Recognition.STT_eng import STTeng

# STTkor model 불러오기
from Voice_Recognition.STT_kor import STTkor

# TTS_kor model 불러오기
from Voice_Recognition.TTS_kor import TTSkor

# face_classification model 불러오기
from Classification.face_classification import resnet34_face
from PIL import Image

# Predict_Ornamental model 불러오기
from Classification.ornamental_predict import Ornamental



def create_app():
    app = Flask(__name__)

    # 기본url에 나오는 것
    @app.route('/')
    def index():
        return '이곳은 오현승의 모듈공간입니다.'

    # body에서 application/json - Raw input으로
    # {"text":"배고프다"} 이런식으로 입력값을 주면 됨.
    @app.route('/ChatBot', methods = ['POST'])
    def chat():
        data = request.get_json()
        text_temp = data['text']
        print("상대방의 말: " , text_temp)

        chatresult = aichatbot(text_temp)
        print("AI의 말: ",chatresult)

        return {'result':chatresult}

    # STT ENG 모델
    # body에서 application/octet-stream으로
    # wav(음성)파일을 전달해주면됨.
    @app.route('/STT_eng', methods=['POST'])
    def sound_to_text_eng():
        # json 파일인지 확인
        # print(request.is_json)

        # bytes파일로 받기
        data = request.data

        # test.wav파일로 저장
        with open(f'test.wav', mode='bx') as f:
            f.write(data)

        result_sound = STTeng(f'test.wav')
        print("음성: " , result_sound)

        # 위에서 저장한 wav파일 삭제
        os.remove('test.wav')

        return  result_sound


    # STT KOR 모델
    # body에서 application/octet-stream으로
    # wav(음성)파일을 전달해주면됨.
    @app.route('/STT_kor', methods=['POST'])
    def sound_to_text_kor():
        # json 파일인지 확인
        # print(request.is_json)

        # bytes파일로 받기
        data = request.data

        # test.wav파일로 저장
        with open(f'test.wav', mode='bx') as f:
            f.write(data)

        result_sound = STTkor(f'test.wav')
        print("음성: ", result_sound)

        # 위에서 저장한 wav파일 삭제
        os.remove('test.wav')

        return result_sound

    @app.route('/EDITH', methods=['POST'])
    def kor_to_chatbot_to_voice():
        # json 파일인지 확인
        # print(request.is_json)

        # bytes파일로 받기
        data = request.data

        # test.wav파일로 저장
        with open(f'test.wav', mode='bx') as f:
            f.write(data)

        # 음성인식
        result_sound = STTkor(f'test.wav')
        print("상대방의 말: ", result_sound)

        # 인식된 text를 chatbot에 넣음
        chatresult = aichatbot(result_sound)
        print("AI의 말: ", chatresult)

        # TTSmodel
        # TTSkor(chatresult)

        # 위에서 저장한 wav파일 삭제
        os.remove('test.wav')

        return chatresult

    @app.route('/predict_face', methods=['POST'])
    def predict_face():

        print("asdasdas")
        # file = request.files['file']
        # file.save('test1.jpg')
        #
        # face_pred = resnet34_face('test1.jpg')
        #
        # os.remove('test1.jpg')


        # 이미지 바로 오픈
        image = request.files['file'].stream
        print(type(image))


        print("fdfdfd")
        face_pred = resnet34_face(image)


        return {'result' : face_pred }

    # http://243114.com/RecievePost

    @app.route('/predict_ornamental', methods=['POST'])
    def predict_Ornamentlal():
        print("asdasdas")
        file = request.files['file']
        file.save('test1.jpg')

        face_pred = Ornamental('test1.jpg')

        shutil.rmtree('C:/Users/HP/Desktop/OHmodule/crop_img')
        # os.remove('C:/Users/HP/Desktop/OHmodule/crop_img')


        return {'result': face_pred}

    return app

