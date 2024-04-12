# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:58:22 2024

@author: chouh
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import whisper
import google.generativeai as genai


app=FastAPI()

class model_input(BaseModel):
    Drive: str()

whisper_audio_model = pickle.load(open('whisper_audio_model.sav','rb'))

@app.get('/audio_feedback')

def audio_feed(input_parameters:model_input):
    
    input_data=input_parameters.json()
    input_dictionary= json.loads(input_data)
    
    dr = input_dictionary['']
    
    input_list=[dr]
    
    pace = whisper_audio_model.predict([input_list])
    return pace


def pace_feedback(pace):
    ideal_lower_bound = 140
    ideal_upper_bound = 160

    feedback = []
    if pace < ideal_lower_bound:
      feedback.append("Your speaking pace is slower than ideal.")
    elif pace > ideal_upper_bound:
      feedback.append("Your speaking pace is faster than ideal.")
    else:
      feedback.append("Your speaking pace is within the ideal range.")
    return feedback
    
   
model1 = whisper.load_model('large')
decode_options={"language":"english"}
text = model1.transcribe(audio_file,word_timestamps= True,verbose=True)

def check_grammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    for match in matches:
        print(match)
        
model = genai.GenerativeModel('gemini-1.0-pro-latest')
response = model.generate_content( str(text))

return response
    