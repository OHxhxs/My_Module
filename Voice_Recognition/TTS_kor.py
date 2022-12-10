'''
google api를 이용한 TTS

pip install playsound
'''

from gtts import gTTS
import os
import time
import playsound



def TTSkor(text):

     tts = gTTS(text=text, lang='ko')

     filename='voice.mp3'
     tts.save(filename)

     print("Answer---")
     playsound.playsound(filename)

     os.remove(filename)

if __name__ == "__main__":
     TTSkor("안녕하세요. 길동씨")