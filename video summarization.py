import streamlit as st
import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi
import nltk
import re
import nltk.corpus 
import sklearn
import transformers
from transformers import BartTokenizer,BartForConditionalGeneration
from gtts import gTTS

link = st.text_input("Enter the link")

if st.button("SUMMARIZE"):
    unique_id = link.split("=")[-1]
    sub = YouTubeTranscriptApi.get_transcript(unique_id)
    subtitle = " ".join([x['text'] for x in sub])
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    input_tensor = tokenizer.encode(subtitle,return_tensors="pt",max_length=512,truncation=True)
    output_tensor = model.generate(input_tensor,max_length=160,min_length=120,length_penalty=2.0,num_beams = 4,early_stopping= True)
    result = tokenizer.decode(output_tensor[0])
    final_res = result.replace("</s>","")
    st.write(final_res)
    myobj = gTTS(text=final_res, lang="en", slow=False)
    myobj.save("welcome.mp3")
    audio_file = open('welcome.mp3', 'rb')
    audio_bytes = audio_file.read()

    st.audio(audio_bytes, format='audio/ogg')
    


