# utils/voice_utils.py

import speech_recognition as sr
from gtts import gTTS

def speech_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language='ta')  # 'ta' for Tamil
        return text
    except sr.UnknownValueError:
        return "மன்னிக்கவும், நான் உங்கள் குரலை புரிந்து கொள்ளவில்லை."
    except sr.RequestError as e:
        return f"சேவை பிழை: {e}"

def text_to_speech(text, output_file_path):
    tts = gTTS(text=text, lang='ta')  # 'ta' for Tamil
    tts.save(output_file_path)
