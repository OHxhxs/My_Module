'''
pip install SpeechRecognition

'''


import speech_recognition as sr

def STTeng(file_path):
  r = sr.Recognizer()

  # american_audio = sr.AudioFile('C:/Users/HP/PycharmProjects/speech_test/' + file_path)
  american_audio = sr.AudioFile(file_path)

  with american_audio as source:
    audio = r.record(source)
  result_sound = r.recognize_google(audio, language='en-US')

  # print(result_sound)

  return result_sound

if __name__ == "__main__":
  STTeng('./Sound_folder/wingardium_leviosa.wav')