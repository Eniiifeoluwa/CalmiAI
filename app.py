import streamlit as st
import os
import time
from huggingface_hub import InferenceClient
from datetime import datetime

# Configuration
MODEL_ID = "Eniiifeoluwa/mental-health-llama2-merged"
API_TOKEN = os.getenv("API_TOKEN")

# Initialize client
if API_TOKEN:
    client = InferenceClient(token=API_TOKEN)
else:
    st.error("⚠️ HF_API_TOKEN environment variable not found. Please set your Hugging Face API token.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Mental Health Support Chatbot",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        margin-bottom: 20px;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 15px;
        margin: 20px 0;
        color: #856404;
    }
    .crisis-info {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 10px;
        padding: 15px;
        margin: 20px 0;
        color: #721c24;
    }
    .user-message {
        text-align: right;
        margin: 10px 0;
    }
    .user-bubble {
        background-color: #dcf8c6;
        padding: 10px 15px;
        border-radius: 18px;
        display: inline-block;
        color: black;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .bot-message {
        text-align: left;
        margin: 10px 0;
    }
    .bot-bubble {
        background-color: #e3f2fd;
        padding: 10px 15px;
        border-radius: 18px;
        display: inline-block;
        color: black;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .timestamp {
        font-size: 0.8em;
        color: #666;
        margin: 5px;
    }
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

# Sidebar with information and settings
with st.sidebar:
    st.header("💡 About This Chatbot")
    st.write("""
    This AI assistant is designed to provide supportive conversations 
    and mental health resources. It uses empathetic language patterns 
    to help you process your thoughts and feelings.
    """)
    
    # User personalization
    st.header("👋 Personalization")
    user_name = st.text_input("Your name (optional):", value=st.session_state.user_name)
    if user_name != st.session_state.user_name:
        st.session_state.user_name = user_name
    
    # Chat statistics
    st.header("📊 Chat Statistics")
    st.write(f"Messages exchanged: {len(st.session_state.history)}")
    
    # Settings
    st.header("⚙️ Settings")
    show_timestamps = st.checkbox("Show message timestamps", value=True)
    max_response_length = st.slider("Max response length", 100, 500, 200)
    
    # Resources
    st.header("🆘 Crisis Resources")
    st.markdown("""
    **If you're in crisis, please reach out:**
    - **National Suicide Prevention Lifeline:** 988
    - **Crisis Text Line:** Text HOME to 741741
    - **International:** befrienders.org
    """)

# Important disclaimer
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

if not st.session_state.show_disclaimer:
    # Helper functions for message display
    def user_message(msg, timestamp=None):
        time_str = ""
        if show_timestamps and timestamp:
            time_str = f'<div class="timestamp">{timestamp}</div>'
        
        greeting = f"Hi {st.session_state.user_name}! " if st.session_state.user_name else ""
        st.markdown(f"""
            <div class="user-message">
                <div class="user-bubble">
                    {msg} 😊
                </div>
                {time_str}
            </div>
        """, unsafe_allow_html=True)

    def bot_message(msg, timestamp=None):
        time_str = ""
        if show_timestamps and timestamp:
            time_str = f'<div class="timestamp">{timestamp}</div>'
        
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
        st.write("")  # Spacing
        send_button = st.button("Send 📤", type="primary")
        clear_button = st.button("Clear Chat 🗑️")

    # Handle sending messages
    if send_button and user_input.strip():
        # Add user message with timestamp
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.history.append({
            'user': user_input, 
            'bot': "...", 
            'timestamp': timestamp,
            'bot_timestamp': None
        })

        with st.spinner("🤔 Thinking..."):
            try:
                # Enhanced prompt with better context
                name_context = f"The user's name is {st.session_state.user_name}. " if st.session_state.user_name else ""
                
                prompt = f"""<s>[INST] <<SYS>>
You are a compassionate and empathetic mental health support assistant. {name_context}Your role is to:
- Listen actively and validate feelings
- Ask thoughtful follow-up questions
- Provide emotional support and coping strategies
- Encourage professional help when appropriate
- Never diagnose or provide medical advice
- Maintain a warm, non-judgmental tone
- Keep responses helpful but concise

Remember: You are not a replacement for professional therapy or medical care.
<</SYS>>

{user_input} [/INST]"""

                response = client.text_generation(
                    prompt,
                    model=MODEL_ID,
                    parameters={
                        "max_new_tokens": max_response_length,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 50,
                        "repetition_penalty": 1.15,
                        "no_repeat_ngram_size": 3,
                        "do_sample": True
                    }
                )

                # Clean up the response
                bot_reply = response.generated_text.split('[/INST]')[-1].strip()
                
                # Remove any unwanted tokens
                unwanted_tokens = ["<s>", "</s>", "<unk>"]
                for token in unwanted_tokens:
                    bot_reply = bot_reply.replace(token, "").strip()

                # Ensure response ends properly
                if bot_reply and not bot_reply.endswith(('.', '!', '?')):
                    bot_reply += '.'

                # Update the last message with the response
                bot_timestamp = datetime.now().strftime("%H:%M")
                st.session_state.history[-1]['bot'] = bot_reply
                st.session_state.history[-1]['bot_timestamp'] = bot_timestamp

            except Exception as e:
                error_msg = "I'm sorry, I'm having trouble connecting right now. Please try again in a moment."
                st.session_state.history[-1]['bot'] = error_msg
                st.session_state.history[-1]['bot_timestamp'] = datetime.now().strftime("%H:%M")
                st.error(f"Error: {str(e)}")

        st.rerun()

    # Handle clearing chat
    if clear_button:
        st.session_state.history = []
        st.success("Chat cleared! Feel free to start a new conversation.")
        st.rerun()

    # Footer with additional resources
    st.markdown("---")
    with st.expander("🔗 Mental Health Resources"):
        st.markdown("""
        **Professional Support:**
        - Find a therapist: psychologytoday.com
        - Mental Health America: mhanational.org
        - NAMI (National Alliance on Mental Illness): nami.org
        
        **Self-Care Tips:**
        - Practice deep breathing exercises
        - Maintain a regular sleep schedule
        - Stay connected with supportive friends/family
        - Engage in physical activity
        - Consider mindfulness or meditation
        """)

    # Quick mood check-in buttons
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
            # Simulate sending the mood message
            st.session_state.history.append({
                'user': message,
                'bot': "...",
                'timestamp': datetime.now().strftime("%H:%M"),
                'bot_timestamp': None
            })
            st.rerun()