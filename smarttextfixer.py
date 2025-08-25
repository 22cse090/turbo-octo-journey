import streamlit as st
from textblob import TextBlob
import language_tool_python

# Set up page
st.set_page_config(page_title="SmartTextFixer Lite", layout="centered")
st.title("⚡ SmartTextFixer")
st.markdown("A fast and lightweight AI tool that corrects spelling and grammar to improve your text accuracy in real-time.")

# Load grammar tool
grammar_tool = language_tool_python.LanguageTool('en-US')

# Functions
def correct_spelling(text):
    return str(TextBlob(text).correct())

def correct_grammar(text):
    return grammar_tool.correct(text)

# Session state for example
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Example input
if st.button("🔁 Try Example"):
    st.session_state.input_text = "he go to the libary everywek and says he liek reading"

# Input box
input_text = st.text_area("✍️ Enter your sentence", value=st.session_state.input_text, height=150)

# Correction button
if st.button("✅ Correct Text"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Correcting..."):
            corrected = correct_spelling(input_text)
            corrected = correct_grammar(corrected)
        st.success("✅ Corrected Output:")
        st.text_area("🔍 Result", value=corrected, height=150)

# Footer
st.markdown("---")
st.markdown(" SmartTextFixer • Created by PRERNA SHARMA   • Fast, lightweight, and offline-ready.")
