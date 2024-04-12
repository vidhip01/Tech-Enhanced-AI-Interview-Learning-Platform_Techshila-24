# Techshila'24 ML Report

## Submission by: Himalaya Bhawan, IIT Roorkee

### Team Members:
- Aishwarya Ahuja
- Palak Jhamnani
- Paridhi Jain
- Shruti Chouhan
- Shoilayee Chaudhuri
- Vidhi Patidar

---

## PROBLEM OVERVIEW:

We aimed to develop an AI-based interviewer capable of asking topic-specific questions, analyzing the user's audio responses, and providing feedback. Additionally, it corrects grammatical errors and offers a reference answer to the user.

---

## DATASET CREATION:

### Text dataset creation:

We scraped PDFs and websites containing interview question answers across various domains. This process resulted in a dataset comprising 3048 data points, structured as follows:

## MODEL OVERVIEW:

### Text Model Development:

We explored various text-generation models like Alpaca Large, Mistralai (Mixtral-8x7B), Llama, and Google Gemma. Due to computational limitations and model sizes, we opted for Llama-2-7b-chat-hf. This choice was driven by its:

- Demonstrably good performance
- Efficient training times
- Ability to deliver results quickly

## Audio Model Development:

### Audio Processing:

The script imports necessary libraries such as `os`, `tqdm`, `torch`, `librosa`, and `gradio` for various tasks including file operations, progress tracking, numerical computations, audio processing, and visualization.

1. **Launching Gradio with Wav2Vec2 Model**: Gradio was launched with a Wav2Vec2 model from Hugging Face for audio recording.

2. **Loading Pre-trained Whisper Model**: We loaded a pre-trained Whisper model, which is a speech recognition model. This model transcribes audio files later in the script.

3. **Calculating Speaking Pace**: Speaking pace (words per minute) was calculated based on the duration of each audio file.

4. **Providing Feedback**: After processing all audio files, the script prints the collected speaking paces and offers feedback to the user by comparing them with the ideal speaking pace (140-160 words per minute).


## Speech-to-Text Conversion:

We used Whisper, a state-of-the-art speech-to-text conversion tool, to transcribe spoken responses into text format. Whisper provides high accuracy and robustness, meeting our project's needs and enabling the system to process spoken inputs for analysis.

## Grammatical Error Detection with LanguageTool:

Following speech-to-text conversion, we used LanguageTool, a powerful grammar checking tool, to identify grammatical errors within the transcribed text. LanguageTool employs advanced algorithms to detect various types of grammatical mistakes, including punctuation errors, spelling errors, and syntactical inconsistencies. By integrating LanguageTool into our pipeline, we ensured that users receive accurate and grammatically correct responses.


## Integration with Gemini AI for Feedback Generation:

We integrated Gemini AI, an advanced natural language processing engine, into our system to provide meaningful feedback to users. Gemini AI analyzes transcribed text, identifies strengths and weaknesses, and generates personalized feedback. Leveraging Gemini AI's capabilities, we offer actionable insights for improvement.

### Feedback Components:

- **Positives**: Highlighting strengths or positive aspects.
- **Negatives**: Identifying areas for improvement or weaknesses.
- **Suggestions for Improvement**: Offering constructive recommendations.
- **Action Words**: Recommending impactful words for clarity and effectiveness in interviews.

  ### Requirements:

Before using this project, ensure that you have the following dependencies installed:

- tqdm
- TensorFlow
- PyTorch
- Gradio
- LanguageTool
- Gemini AI
- fastapi
- uvicorn
- pydantic
- scikit-learn
- requests
- os
- librosa
- whisper
- numpy
- pytube
- pathlib
- textwrap
- google.generativeai
- IPython
- transformers
- pefft
- trl
- locale
- loRA
- QLora
  

You can install the required dependencies by running:

```bash
pip install -r requirements.txt



