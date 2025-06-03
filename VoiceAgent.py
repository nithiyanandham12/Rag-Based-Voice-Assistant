import streamlit as st
import speech_recognition as sr
import pyttsx3
import requests
import os
from dotenv import load_dotenv
import threading
import time
from io import BytesIO
import tempfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import re
from collections import Counter
from gtts import gTTS
import pygame
import base64
from collections import defaultdict
import math
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Initialize pygame mixer with error handling
try:
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception as e:
    PYGAME_AVAILABLE = False
    print("Warning: pygame audio not available. Using alternative audio playback.")

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Voice Bot with Watsonx",
    page_icon="🎙️",
    layout="wide"
)

bot_name = "ava"

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False
if 'bearer_token' not in st.session_state:
    st.session_state.bearer_token = None
if 'last_response' not in st.session_state:
    st.session_state.last_response = ""
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'pdf_loaded' not in st.session_state:
    st.session_state.pdf_loaded = False

def load_and_process_pdf(pdf_file):
    """Load and process PDF file for RAG (local, using FAISS)"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(pages)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        os.unlink(tmp_path)
        return vector_store
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def get_rag_response(query, vector_store):
    """Get response using local RAG (returns most relevant PDF chunk(s))"""
    try:
        docs_and_scores = vector_store.similarity_search_with_score(query, k=2)
        if not docs_and_scores:
            return "Sorry, I could not find relevant information in the PDF."
        response = "\n---\n".join([doc.page_content for doc, score in docs_and_scores])
        return response
    except Exception as e:
        return f"Error in RAG response: {str(e)}"

def listen_for_speech():
    """Speech recognition using browser's Web Speech API"""
    st.info("🎤 Click the microphone button to start speaking!")
    
    # Create a microphone input component
    audio_bytes = st.audio_recorder(
        text="Click to record",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_name="microphone",
        icon_size="2x",
    )
    
    if audio_bytes:
        # Save the audio bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        try:
            # Use speech recognition on the saved audio file
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_path) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio, language='en-US')
                return text
        except sr.UnknownValueError:
            st.error("Could not understand audio. Please try speaking again.")
            return "Could not understand audio"
        except sr.RequestError as e:
            st.error(f"Error with speech recognition service: {e}")
            return f"Error with speech recognition service: {e}"
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    return None

def speak_text(text):
    """Convert text to speech using gTTS"""
    try:
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_filename = temp_file.name
        
        # Generate speech using gTTS
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(temp_filename)
        
        # Create an audio player using Streamlit's audio component
        with open(temp_filename, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/mp3')
            
        # Clean up the temporary file
        try:
            os.unlink(temp_filename)
        except:
            pass
            
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")
        # Fallback to pyttsx3 if gTTS fails
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e2:
            st.error(f"Fallback TTS also failed: {e2}")

def process_voice_input():
    """Process voice input"""
    if not st.session_state.bearer_token:
        st.error("Please authenticate first!")
        return
    
    if not st.session_state.pdf_loaded:
        st.error("Please upload a PDF document first!")
        return
    
    while st.session_state.continuous_mode:
        try:
            # Listen for speech
            user_text = listen_for_speech()
            
            # Check if the text is too short
            if len(user_text.split()) < 3 and not any(error in user_text for error in ["Error", "Timeout", "Could not"]):
                st.warning("Speech might have been cut off. Please try speaking again.")
                continue
            
            if user_text and not any(error in user_text for error in ["Error", "Timeout", "Could not"]):
                st.success(f"📝 **You said:** {user_text}")
                
                # Add user input to conversation history
                st.session_state.conversation_history.append(("user", user_text))
                
                # Get RAG response
                with st.spinner("Getting response from PDF..."):
                    ai_response = get_rag_response(user_text, st.session_state.vector_store)
                
                if ai_response and not ai_response.startswith("Error"):
                    # Add AI response to conversation history
                    st.session_state.conversation_history.append(("assistant", ai_response))
                    st.session_state.last_response = ai_response
                    
                    st.success(f"🤖 **AI Response:** {ai_response}")
                    
                    # Speak the response
                    with st.spinner("Speaking response..."):
                        speak_text(ai_response)
                else:
                    st.error(f"AI Error: {ai_response}")
            else:
                continue
                
        except Exception as e:
            st.error(f"Error in voice processing: {str(e)}")
            continue

def get_conversation_summary(conversation_history):
    """Generate a summary of the conversation using Watsonx"""
    if not conversation_history:
        return "No conversation to summarize."
    
    # Format conversation for summary
    conversation_text = "\n".join([f"{role}: {text}" for role, text in conversation_history])
    
    # Create summary prompt
    summary_prompt = f"""Please provide a concise summary of the following conversation:

{conversation_text}

Summary:"""
    
    # Get summary from Watsonx
    try:
        summary = get_watsonx_response(
            [],  # Empty history for summary
            summary_prompt,
            st.session_state.bearer_token
        )
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def send_to_slack(summary, webhook_url, bot_name="Ava"):
    """Send conversation summary to Slack for human agent review"""
    try:
        # Format the message
        message = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"📬 Message from {bot_name} – Conversation Summary",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"Hello team! :wave:\n\n"
                            f"I just wrapped up a conversation with a customer. Here's a summary for your review:"
                        )
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"> {summary.replace(chr(10), chr(10) + '> ')}"
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"_Generated on {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')} by {bot_name}_"
                        }
                    ]
                }
            ]
        }

        # Send to Slack
        response = requests.post(webhook_url, json=message)
        response.raise_for_status()
        return True

    except Exception as e:
        st.error(f"Error sending to Slack: {str(e)}")
        return False

def send_summary_email(summary, recipient_email):
    """Send conversation summary via email and Slack"""
    try:
        # Get email configuration from environment variables
        sender_email = os.getenv("EMAIL_SENDER")
        email_password = os.getenv("EMAIL_PASSWORD")
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")

        if not all([sender_email, email_password]):
            return "Email configuration missing. Please set EMAIL_SENDER and EMAIL_PASSWORD in .env file."

        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"{bot_name} – Your Voice Conversation Summary • {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"

        # Add summary to email body
        body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; color: #333;">
                <h2 style="color: #4B0082;">Hi there! I'm {bot_name} 👋</h2>
                <p>I've put together a quick summary of our recent conversation. Here's what we discussed:</p>
                <div style="background-color: #f0f0f5; padding: 15px; border-left: 5px solid #4B0082; border-radius: 6px; margin: 20px 0;">
                    {summary}
                </div>
                <p>If anything feels off or you'd like me to clarify more, I'm always here to help!</p>
                <p style="margin-top: 30px;">Chat recorded on <strong>{datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}</strong></p>
                <p>With warm regards,</p>
                <p style="font-size: 16px; font-weight: bold;">{bot_name}<br>
                <span style="font-size: 14px; font-weight: normal;">Your Voice Companion</span></p>
            </body>
        </html>
        """

        msg.attach(MIMEText(body, 'html'))

        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, email_password)
            server.send_message(msg)

        # Send to Slack if webhook URL is configured
        if slack_webhook_url:
            slack_success = send_to_slack(summary, slack_webhook_url)
            if slack_success:
                return "Summary sent successfully to email and Slack!"
            else:
                return "Summary sent to email but failed to send to Slack."
        
        return "Summary sent successfully to email!"
    except Exception as e:
        return f"Error sending summary: {str(e)}"

def get_bearer_token(api_key):
    """Get bearer token for Watsonx API authentication"""
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"apikey={api_key}&grant_type=urn:ibm:params:oauth:grant-type:apikey"

    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        st.error(f"Failed to retrieve access token: {response.text}")
        return None

def clean_ai_response(response_text):
    """Clean the AI response by removing template tags and unwanted text"""
    if not response_text:
        return response_text
    
    # Remove common template tags
    unwanted_patterns = [
        "assistant<|end_header_id|>",
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "**",
        "assistant<|end_header_id|>\n\n",
        "assistant<|end_header_id|>\n",
    ]
    
    cleaned_response = response_text
    for pattern in unwanted_patterns:
        cleaned_response = cleaned_response.replace(pattern, "")
    
    # Remove leading/trailing whitespace and newlines
    cleaned_response = cleaned_response.strip()
    
    return cleaned_response

def get_watsonx_response(history, user_input, bearer_token):
    """Get response from Watsonx API"""
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {bearer_token}"
    }

    # Construct the conversation history
    conversation = "".join(
        f"<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>\n" 
        for role, text in history
    )
    
    conversation += f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>\n"

    payload = {
        "input": conversation,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 8100,
            "min_new_tokens": 0,
            "stop_sequences": [],
            "repetition_penalty": 1
        },
        "model_id": "meta-llama/llama-3-3-70b-instruct",
        "project_id": os.getenv("PROJECT_ID")
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        response_data = response.json()
        if "results" in response_data and response_data["results"]:
            raw_response = response_data["results"][0]["generated_text"]
            return clean_ai_response(raw_response)
        else:
            return "Error: 'generated_text' not found in the response."
    else:
        return f"Error: Failed to fetch response from Watsonx.ai. Status code: {response.status_code}"

# Main UI
st.title("🎙️ Voice Bot with Watsonx LLM")
st.markdown("### Voice Assistant")
st.markdown("---")

# PDF Upload Section
st.header("📄 PDF Document")
uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        vector_store = load_and_process_pdf(uploaded_file)
        if vector_store:
            st.session_state.vector_store = vector_store
            st.session_state.pdf_loaded = True
            st.success("✅ PDF processed successfully!")

# Auto-authentication on app start
if not st.session_state.bearer_token:
    api_key = os.getenv("API_KEY")
    project_id = os.getenv("PROJECT_ID")
    
    if api_key and project_id:
        with st.spinner("Authenticating with Watsonx..."):
            token = get_bearer_token(api_key)
            if token:
                st.session_state.bearer_token = token
                st.success("✅ Authentication successful!")
            else:
                st.error("❌ Authentication failed! Please check your API_KEY in .env file")
    else:
        st.error("❌ Missing API_KEY or PROJECT_ID in environment variables. Please check your .env file.")

st.markdown("---")

# Voice interaction section
st.header("🎤 Voice Interaction")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎙️ Start Continuous Voice Chat", disabled=not st.session_state.bearer_token or not st.session_state.pdf_loaded):
        st.session_state.continuous_mode = True
        process_voice_input()

with col2:
    if st.button("⏹️ Stop Voice Chat"):
        st.session_state.continuous_mode = False
        st.success("Voice chat stopped!")

with col3:
    if st.button("🗑️ Clear Conversation"):
        st.session_state.conversation_history = []
        st.session_state.last_response = ""
        st.success("Conversation cleared!")

st.markdown("---")

# Conversation history display
st.header("📝 Conversation History")
if st.session_state.conversation_history:
    for i, (role, text) in enumerate(st.session_state.conversation_history):
        if role == "user":
            st.markdown(f"**👤 You:** {text}")
        else:
            st.markdown(f"**🤖 Assistant:** {text}")
            # Add individual speak button for each response
            if st.button(f"🔊 Speak", key=f"speak_{i}"):
                speak_text(text)
else:
    st.info("No conversation yet. Start by clicking 'Start Voice Chat' or typing a message.")

# Add this after the conversation history display section
st.markdown("---")
st.header("📊 Conversation Summary")

# Replace the email input field with hardcoded email
email_address = "ananthananth881@gmail.com"  # Replace this with your actual email address

# Add a button to generate and send summary
col1, col2 = st.columns(2)
with col1:
    if st.button("Generate Summary"):
        with st.spinner("Generating conversation summary..."):
            summary = get_conversation_summary(st.session_state.conversation_history)
            st.markdown("### Summary")
            st.markdown(summary)
            
            # Store summary in session state
            st.session_state.last_summary = summary

with col2:
    if st.button("Send Summary via Email"):
        if not st.session_state.get('last_summary'):
            st.warning("Please generate a summary first!")
        else:
            with st.spinner("Sending summary via email..."):
                result = send_summary_email(st.session_state.last_summary, email_address)
                if "successfully" in result:
                    st.success(result)
                else:
                    st.error(result)

# Add automatic summary every 5 messages
if len(st.session_state.conversation_history) > 0 and len(st.session_state.conversation_history) % 5 == 0:
    with st.spinner("Generating periodic summary..."):
        summary = get_conversation_summary(st.session_state.conversation_history)
        st.markdown("### Periodic Summary")
        st.markdown(summary)
        
        # Store summary in session state
        st.session_state.last_summary = summary
        
        # If email is provided, offer to send the periodic summary
        if email_address:
            if st.button("Send Periodic Summary via Email"):
                with st.spinner("Sending periodic summary via email..."):
                    result = send_summary_email(summary, email_address)
                    if "successfully" in result:
                        st.success(result)
                    else:
                        st.error(result)

# Status indicators
st.sidebar.header("📊 Status")
st.sidebar.success("✅ Ready" if st.session_state.bearer_token else "❌ Not Authenticated")
st.sidebar.info(f"💬 Messages: {len(st.session_state.conversation_history)}")
