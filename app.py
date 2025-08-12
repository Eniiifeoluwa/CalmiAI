import streamlit as st
import os
from huggingface_hub import InferenceClient

MODEL_ID = "Eniiifeoluwa/mental-health-llama2-merged"

API_TOKEN = os.getenv("HF_API_TOKEN")


client = InferenceClient(repo_id=MODEL_ID, token=API_TOKEN)

st.title("🤗 Mental Health Chatbot")

if 'history' not in st.session_state:
    st.session_state.history = []

def user_message(msg):
    st.markdown(f"""
        <div style="text-align: right; margin: 10px;">
            <span style="background-color: #dcf8c6; padding: 10px; border-radius: 10px; display: inline-block; color: black; max-width: 60%;">
                {msg} 😊
            </span>
        </div>
    """, unsafe_allow_html=True)

def bot_message(msg):
    st.markdown(f"""
        <div style="text-align: left; margin: 10px;">
            <span style="background-color: #f1f0f0; padding: 10px; border-radius: 10px; display: inline-block; color: black; max-width: 60%;">
                {msg} 🤖
            </span>
        </div>
    """, unsafe_allow_html=True)

# Show chat history
for chat in st.session_state.history:
    user_message(chat['user'])
    bot_message(chat['bot'])

user_input = st.text_area("How are you feeling today?", height=100)

if st.button("Send") and user_input.strip():
    st.session_state.history.append({'user': user_input, 'bot': "..."})

    with st.spinner("Thinking..."):
        # Optionally add your prompt formatting here if needed
        prompt = f"<s>[INST] <<SYS>>\nYou are a helpful and empathetic mental health assistant.\n<</SYS>>\n\n{user_input} [/INST]"
        
        response = client.text_generation(
            prompt,
            max_new_tokens=200,
            temperature=0.5,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
        
        bot_reply = response[0]['generated_text'].split('[/INST]')[-1].strip()
        
        # Remove trailing <s> if exists
        if bot_reply.endswith("<s>"):
            bot_reply = bot_reply[:-3].strip()
        
        # Optionally truncate to first 5 sentences
        if '.' in bot_reply:
            sentences = bot_reply.split('.')
            bot_reply = '.'.join(sentences[:5]).strip()
            if not bot_reply.endswith('.'):
                bot_reply += '.'

    st.session_state.history[-1]['bot'] = bot_reply
    st.experimental_rerun()

if st.button("Clear Chat"):
    st.session_state.history = []
    st.experimental_rerun()
