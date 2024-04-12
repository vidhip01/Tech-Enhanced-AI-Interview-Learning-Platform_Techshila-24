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

We explored various text-generation models such as Alpaca Large, Mistralai (Mixtral-8x7B), Llama, and Google Gemma. However, due to computational limitations and the large size of some models, we opted for Llama-2-7b-chat-hf. We made this decision based on the following factors:

- **Demonstrably good performance**: Llama-2-7b-chat-hf showed promising results in text generation tasks.
- **Efficient training times**: Training the Llama model was relatively faster compared to other models we considered.
- **Ability to deliver results quickly**: Llama-2-7b-chat-hf provided timely outputs, aligning with our project's requirements.

### Audio Model Development:

#### Audio Processing:

The script for audio processing involves importing necessary libraries such as `os`, `tqdm`, `torch`, `librosa`, and `gradio`, which serve various purposes including file operations, progress tracking, numerical computations, audio processing, and visualization.

1. **Launching Gradio with a Wav2Vec2 model**: We utilized Gradio to launch a user interface for recording audio, integrating it with the Wav2Vec2 model from Hugging Face.
  
2. **Loading Pre-trained Whisper Model**: We loaded a pre-trained Whisper model, a speech recognition model where 'base' denotes a large pre-trained model variant. This model is employed to transcribe audio files later in the script.

3. **Calculating Speaking Pace**: We calculated the speaking pace (words per minute) based on the duration of each audio file. This metric helps in analyzing the speed of speech. 

4. **Providing Feedback**: After processing all audio files, the script prints the collected speaking paces and offers feedback to the user by comparing them with the ideal speaking pace, typically ranging from 140 to 160 words per minute (wpm).


## Speech-to-Text Conversion:

We employed Whisper, a state-of-the-art speech-to-text conversion tool, to transcribe spoken responses into text format. Whisper's high accuracy and robustness make it suitable for our project's requirements. This step was crucial in enabling our system to process spoken inputs for further analysis.

## Grammatical Error Detection with LanguageTool:

Following the speech-to-text conversion, we integrated LanguageTool into our pipeline for grammatical error detection. LanguageTool is a powerful grammar checking tool that employs advanced algorithms to identify various types of grammatical mistakes, including punctuation errors, spelling errors, and syntactical inconsistencies. By leveraging LanguageTool, we ensure that the user receives accurate and grammatically correct responses, enhancing the overall quality and professionalism of the interaction with our AI-based interviewer.

## Integration with Gemini AI for Feedback Generation:

To provide meaningful feedback to users, we integrated Gemini AI, an advanced natural language processing engine, into our system. Gemini AI analyzes the transcribed text, identifies strengths and weaknesses, and generates personalized feedback. By leveraging Gemini AI's capabilities, we offer actionable insights for improvement.

### Feedback Components:
- **Positives**: Highlighting strengths or positive aspects.
- **Negatives**: Identifying areas for improvement.
- **Suggestions for Improvement**: Offering constructive recommendations.
- **Action Words**: Recommending impactful words for clarity and effectiveness.


