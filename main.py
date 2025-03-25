import streamlit as st
from transformers import pipeline
from gtts import gTTS
import os
import base64

# Set page configuration
st.set_page_config(
    page_title="Voice-Enabled AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_model():
    """Load the text generation pipeline (cached to avoid reloading)"""
    # Options for different models:
    # 1. Question-answering model
    qa_model = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
    
    # 2. Text generation model (alternative)
    text_gen_model = pipeline('text-generation', model='gpt2')
    
    return {
        "qa": qa_model,
        "text_gen": text_gen_model
    }

def generate_response(user_input, models, model_type="qa"):
    """Generate a response using the selected language model"""
    try:
        if model_type == "qa":
            # For QA models, we need a context. We'll use the question itself as minimal context
            # This works for simple factual questions but is limited
            context = "I am an AI assistant. " + user_input
            response = models["qa"](question=user_input, context=context)
            return response['answer']
        else:
            # For text generation models
            prompt = f"Question: {user_input}\nAnswer:"
            response = models["text_gen"](
                prompt,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            generated_text = response[0]['generated_text']
            new_text = generated_text[len(prompt):].strip()
            
            if not new_text:
                return "I'm not sure how to respond to that. Can you ask something else?"
            return new_text
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I encountered an error while processing your request. Please try again."

def text_to_speech(text):
    """Convert text to speech and return the audio file as base64"""
    tts = gTTS(text=text, lang='en', slow=False)
    audio_file = "response.mp3"
    tts.save(audio_file)
    
    # Convert to base64 for embedding in HTML
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    
    # Remove the temporary file
    os.remove(audio_file)
    
    return audio_base64

def main():
    st.title("🤖 Voice-Enabled AI Assistant")
    st.markdown("Ask me anything, and I'll respond with voice!")
    
    # Load models
    with st.spinner("Loading AI models..."):
        models = load_model()
    
    # Model selection
    model_type = st.sidebar.radio(
        "Select Model Type",
        ["Question Answering (DistilBERT)", "Text Generation (GPT-2)"]
    )
    
    # Map selection to model type
    model_key = "qa" if "Question Answering" in model_type else "text_gen"
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Get user input
    user_input = st.chat_input("Ask a question...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(user_input, models, model_key)
                st.write(response)
                
                # Convert response to speech
                with st.spinner("Generating voice..."):
                    audio_base64 = text_to_speech(response)
                    
                    # Display audio player
                    st.markdown(
                        f'<audio autoplay controls><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>',
                        unsafe_allow_html=True
                    )
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
