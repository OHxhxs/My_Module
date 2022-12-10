'''
카카오 api를 이용한 STT

pip install python-decouple
# SECRET_KEY = '3f4fb64c93e4e5ea31cf222cf03642ae'
실패

그래서 구글 api 활용
'''

import speech_recognition as sr

def STTkor(file_path):
  r = sr.Recognizer()

  # american_audio = sr.AudioFile('C:/Users/HP/PycharmProjects/speech_test/' + file_path)
  american_audio = sr.AudioFile(file_path)

  with american_audio as source:
    audio = r.record(source)
  result_sound = r.recognize_google(audio, language='ko-KR')

  # print(result_sound)

  return result_sound

if __name__ == "__main__":
  STTkor('./Sound_folder/저사람.wav')