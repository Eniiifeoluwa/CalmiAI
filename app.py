import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import accelerate
import os

# Model name
MODEL_NAME = "Eniiifeoluwa/mental-gemma"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Cached model loader
@st.cache_resource(show_spinner="Loading model... please wait ⏳")
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",               # auto-distribute across GPU/CPU
        trust_remote_code=True,
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.eval()
    return model, tokenizer

# ✅ Response generator
def generate_response(prompt, model, tokenizer, max_new_tokens=200, temperature=0.7):
    formatted_prompt = f"""
    instruction: You are a professional, empathetic mental health assistant.  
    Be concise, specific, and supportive. Show empathy and provide the best advice.  
    Avoid hallucinations. Always answer as if you are Samuel, a caring listener.  
    The user said: {prompt}  
    output:
    """

    enc = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)

    try:
        with torch.no_grad():
            sampled_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.8,
                top_k=40,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        sampled_text = tokenizer.decode(sampled_ids[0], skip_special_tokens=True)

        # ✅ Extract only output part
        if "output:" in sampled_text:
            answer = sampled_text.split("output:")[-1].strip()
        else:
            answer = sampled_text.strip()

        # Ensure punctuation spacing
        answer = answer.replace(".", ". ").replace("?", "? ").replace("!", "! ")
        answer = " ".join(answer.split())  # normalize spaces

        return answer.strip()

    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"

# Load model once
try:
    model, tokenizer = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    model_loaded = False

# Streamlit Page Configuration
st.set_page_config(
    page_title="Mental Health Support Chatbot",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔥 Streamlit UI styling
st.markdown("""
<style>
    .main-header { text-align: center; color: #2E8B57; margin-bottom: 20px; }
    .disclaimer { background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px; padding: 15px; margin: 20px 0; color: #856404; }
    .crisis-info { background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 10px; padding: 15px; margin: 20px 0; color: #721c24; }
    .user-message { text-align: right; margin: 10px 0; }
    .user-bubble { background-color: #dcf8c6; padding: 10px 15px; border-radius: 18px; display: inline-block; color: black; max-width: 70%; word-wrap: break-word; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .bot-message { text-align: left; margin: 10px 0; }
    .bot-bubble { background-color: #e3f2fd; padding: 10px 15px; border-radius: 18px; display: inline-block; color: black; max-width: 70%; word-wrap: break-word; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .timestamp { font-size: 0.8em; color: #666; margin: 5px; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'show_disclaimer' not in st.session_state:
    st.session_state.show_disclaimer = True
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""

# Header
st.markdown('<h1 class="main-header">🤗 Mental Health Support Chatbot</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("💡 About This Chatbot")
    st.write("""
    This AI assistant is designed to provide supportive conversations 
    and mental health resources. It uses empathetic language patterns 
    to help you process your thoughts and feelings.
    """)
    st.header("👋 Personalization")
    user_name = st.text_input("Your name (optional):", value=st.session_state.user_name)
    if user_name != st.session_state.user_name:
        st.session_state.user_name = user_name
    st.header("📊 Chat Statistics")
    st.write(f"Messages exchanged: {len(st.session_state.history)}")
    st.header("⚙️ Settings")
    show_timestamps = st.checkbox("Show message timestamps", value=True)
    st.header("🆘 Crisis Resources")
    st.markdown("""
    **If you're in crisis, please reach out:**
    - [Crisis Support](https://www.betterhelp.com/get-started/)
    - [Mental Health Resources](https://www.psychologytoday.com/us)
    - [Online Counseling](https://www.talkspace.com/)
    """)

# Disclaimer
if st.session_state.show_disclaimer:
    st.markdown("""
    <div class="disclaimer">
        <h4>⚠️ Important Disclaimer</h4>
        <p>This chatbot is an AI assistant designed to provide supportive conversations 
        and general mental health information. It is <strong>not a replacement for professional 
        mental health services</strong>, therapy, or medical advice.</p>
        <p>If you're experiencing a mental health crisis or having thoughts of self-harm, 
        please contact a mental health professional, your doctor, or emergency services immediately.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("I understand - Continue"):
        st.session_state.show_disclaimer = False
        st.rerun()

# Chat functionality
if not st.session_state.show_disclaimer and model_loaded:

    def user_message(msg, timestamp=None):
        time_str = f'<div class="timestamp">{timestamp}</div>' if show_timestamps and timestamp else ""
        st.markdown(f"""
            <div class="user-message">
                <div class="user-bubble">
                    {msg} 😊
                </div>
                {time_str}
            </div>
        """, unsafe_allow_html=True)

    def bot_message(msg, timestamp=None):
        time_str = f'<div class="timestamp">{timestamp}</div>' if show_timestamps and timestamp else ""
        st.markdown(f"""
            <div class="bot-message">
                <div class="bot-bubble">
                    🤖 {msg}
                </div>
                {time_str}
            </div>
        """, unsafe_allow_html=True)

    # Display chat history
    for chat in st.session_state.history:
        user_message(chat['user'], chat.get('timestamp'))
        bot_message(chat['bot'], chat.get('bot_timestamp'))

    # Input section
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_area(
            "How are you feeling today? Share what's on your mind...",
            height=100,
            placeholder="Type your message here...",
            key="user_input"
        )
    with col2:
        st.write("")
        send_button = st.button("Send 📤", type="primary")
        clear_button = st.button("Clear Chat 🗑️")

    if send_button and st.session_state.user_input.strip():
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.history.append({
            'user': st.session_state.user_input,
            'bot': "...",
            'timestamp': timestamp,
            'bot_timestamp': None
        })
        with st.spinner("🤔 Thinking..."):
            bot_reply = generate_response(st.session_state.user_input, model, tokenizer)
            bot_timestamp = datetime.now().strftime("%H:%M")
            st.session_state.history[-1]['bot'] = bot_reply
            st.session_state.history[-1]['bot_timestamp'] = bot_timestamp

        # ✅ Clear the input box after sending
        st.session_state.user_input = ""
        st.rerun()

    if clear_button:
        st.session_state.history = []
        st.success("Chat cleared! Feel free to start a new conversation.")
        st.rerun()

    # Footer
    st.markdown("---")
    with st.expander("🔗 Mental Health Resources"):
        st.markdown("""
        **Professional Support:**
        - [Find a therapist](https://www.psychologytoday.com)
        - [Mental Health America](https://www.mhanational.org)
        - [NAMI (National Alliance on Mental Illness)](https://www.nami.org)
        - [BetterHelp Online Therapy](https://www.betterhelp.com)
        - [Talkspace](https://www.talkspace.com)

        **Self-Care Tips:**
        - Practice deep breathing exercises
        - Maintain a regular sleep schedule
        - Stay connected with supportive friends/family
        - Engage in physical activity
        - Consider mindfulness or meditation
        """)

    # Quick mood check-in
    st.markdown("### Quick Check-in")
    col1, col2, col3, col4 = st.columns(4)
    moods = {
        "😊 Good": "I'm feeling good today!",
        "😔 Down": "I'm feeling down today and could use some support.",
        "😰 Anxious": "I'm feeling anxious and overwhelmed.",
        "😴 Tired": "I'm feeling really tired and drained."
    }
    for i, (mood, message) in enumerate(moods.items()):
        col = [col1, col2, col3, col4][i]
        if col.button(mood):
            st.session_state.history.append({
                'user': message,
                'bot': "...",
                'timestamp': datetime.now().strftime("%H:%M"),
                'bot_timestamp': None
            })
            st.rerun()

elif not model_loaded:
    st.error("Model could not be loaded. Please check your model configuration and try again.")
