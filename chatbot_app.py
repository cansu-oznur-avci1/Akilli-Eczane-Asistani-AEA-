from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from main import run_agent, run_agent_stream, generate_chat_title
import os
import pandas as pd
import json
import base64
import uuid
from utils import history_manager as hm
from utils.formatting_utils import format_assistant_response
from utils.i18n import get_text

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return ""


st.set_page_config(page_title="Akıllı Eczacı Asistanı", page_icon="💊", layout="wide")

if "lang" not in st.session_state:
    st.session_state.lang = "tr"
if "dark_theme" not in st.session_state:
    st.session_state.dark_theme = True

# Dynamic CSS based on theme
theme_css = """
<style>
/* Modern typography */
html, body, [class*="css"]  {
    font-family: 'Inter', 'Roboto', sans-serif;
}

/* Chat message container styling */
.stChatMessage {
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

/* Force consistent text and header sizes in all chat messages */
.stChatMessage h1, .stChatMessage h2, .stChatMessage h3, .stChatMessage h4 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.5rem !important;
    line-height: 1.4 !important;
}

.stChatMessage p, .stChatMessage li, .stChatMessage span {
    font-size: 1rem !important;
    line-height: 1.6 !important;
}

/* Sidebar styling improvements */
.stButton>button {
    border-radius: 8px;
    transition: all 0.2s;
    font-size: 1rem !important;
}

/* Unified settings buttons style */
.settings-bar-container {
    margin-bottom: 0.2rem !important;
}
.settings-bar-container button {
    border-radius: 20px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 0.3rem 0.8rem !important;
    height: 2.2rem !important;
    transition: all 0.2s ease-in-out !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Sidebar divider spacing adjustment */
[data-testid="stSidebar"] hr {
    margin-top: 0.2rem !important;
    margin-bottom: 0.8rem !important;
    opacity: 0.5 !important;
}
"""

if st.session_state.dark_theme:
    theme_css += """
    [data-testid="stAppViewContainer"] { background-color: #121212 !important; color: #E0E0E0 !important; }
    [data-testid="stSidebar"] { background-color: #1E1E1E !important; color: #E0E0E0 !important; border-right: 1px solid #333; }
    [data-testid="stHeader"] { background-color: #121212 !important; color: #E0E0E0 !important; }
    .stMarkdown, p, h1, h2, h3, h4, span, label, [data-testid="stWidgetLabel"] p, [data-testid="stExpander"] summary span { color: #E0E0E0 !important; }
    
    /* Fix buttons in dark mode */
    .stButton>button, [data-testid="stFormSubmitButton"]>button { background-color: #2D2D2D !important; color: #E0E0E0 !important; border: 1px solid #444 !important; }
    
    /* Fix file uploader */
    [data-testid="stFileUploaderDropzone"] { background-color: #2D2D2D !important; border: 1px dashed #444 !important; }
    [data-testid="stFileUploaderDropzone"] * { color: #E0E0E0 !important; }
    [data-testid="stFileUploaderDropzone"] button { background-color: #1E1E1E !important; color: #E0E0E0 !important; border: 1px solid #555 !important; }
    [data-testid="stFileUploader"] * { color: #E0E0E0 !important; }
    
    /* Fix toggle background */
    /* Fix toggle background & active state in dark mode */
    [data-testid="stToggle"] [data-baseweb="checkbox"] > div:first-of-type {
        background-color: #333333 !important;
        border: 1px solid #555555 !important;
    }
    [data-testid="stToggle"] [data-baseweb="checkbox"]:has(input:checked) > div:first-of-type {
        background-color: #ff4b4b !important;
        border-color: #ff4b4b !important;
    }
    [data-testid="stToggle"] [data-baseweb="checkbox"]:has(input:checked) > div:first-of-type > div {
        background-color: #FFFFFF !important;
    }
    [data-testid="stToggle"] [data-baseweb="checkbox"] p,
    [data-testid="stToggle"] [data-baseweb="checkbox"] span {
        color: #E0E0E0 !important;
    }
    
    /* Fix chat input area in dark mode */
    [data-testid="stBottom"], [data-testid="stBottom"] > div { background-color: #121212 !important; }
    [data-testid="stBottomBlockContainer"] { background-color: transparent !important; }
    
    .stChatInputContainer { background-color: transparent !important; border: none !important; }
    
    [data-testid="stChatInput"] { 
        background-color: #2D2D2D !important; 
        border: 1px solid #444444 !important; 
        border-radius: 24px !important; 
        overflow: hidden !important;
    }
    
    [data-testid="stChatInput"] * { 
        background-color: transparent !important; 
        color: #FFFFFF !important; 
        caret-color: #FFFFFF !important;
        border: none !important;
    }
    
    [data-testid="stChatInput"] textarea::placeholder { color: #AAAAAA !important; }
    
    /* Fix status widget in dark mode */
    [data-testid="stStatusWidget"] { background-color: transparent !important; border: none !important; }
    [data-testid="stStatusWidget"] * { background-color: transparent !important; color: #E0E0E0 !important; }
    
    .stChatMessage { 
        background-color: #1E1E1E !important; 
        border: 1px solid #3f444d !important; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }
    
    /* Sidebar adjustments for dark mode */
    [data-testid="stSidebar"] h2 { color: #8F9CAE !important; }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        color: #A0AEC0 !important;
        background-color: transparent !important;
        border: 1px solid transparent !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background-color: rgba(255, 75, 75, 0.05) !important;
        color: #ff4b4b !important;
        border: 1px solid #ff4b4b !important;
        box-shadow: 0 0 8px rgba(255, 75, 75, 0.25) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: rgba(255, 255, 255, 0.12) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background-color: rgba(255, 75, 75, 0.08) !important;
        color: #ff4b4b !important;
        border: 1px solid #ff4b4b !important;
        box-shadow: 0 0 8px rgba(255, 75, 75, 0.25) !important;
    }
    [data-testid="stSidebar"] [data-testid="column"] .stButton > button {
        color: #E53E3E !important;
        opacity: 0.6;
    }
    [data-testid="stSidebar"] [data-testid="column"] .stButton > button:hover {
        opacity: 1 !important;
        background-color: rgba(229, 62, 62, 0.15) !important;
        color: #FC8181 !important;
    }
    
    /* Settings buttons dark mode styling */
    .settings-bar-container button {
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #A0AEC0 !important;
        transition: all 0.2s ease-in-out !important;
    }
    /* Settings buttons red border hover in dark mode (sibling columns selector) */
    div.settings-bar-container + div .stButton button:hover,
    div.element-container:has(div.settings-bar-container) + div.element-container .stButton button:hover {
        border-color: #ff4b4b !important;
        color: #ff4b4b !important;
        box-shadow: 0 0 8px rgba(255, 75, 75, 0.25) !important;
        transform: translateY(-1px) !important;
        background-color: rgba(255, 75, 75, 0.05) !important;
    }

    /* Global button hover red borders in dark mode */
    .stButton>button:hover, [data-testid="stFormSubmitButton"]>button:hover {
        border-color: #ff4b4b !important;
        color: #ff4b4b !important;
        box-shadow: 0 0 8px rgba(255, 75, 75, 0.25) !important;
        transform: translateY(-1px) !important;
    }

    /* Dark Mode Settings Modal - Solid Card with Transparent Backdrop Overlay */
    div[data-baseweb="modal"], div[data-baseweb="modal"] > div:first-child {
        background-color: transparent !important;
        background: transparent !important;
        backdrop-filter: none !important;
    }
    div[role="dialog"], [data-testid="stModal"] {
        background-color: #121212 !important;
        border: 1px solid #333333 !important;
        box-shadow: 0px 8px 32px rgba(0, 0, 0, 0.6) !important;
        border-radius: 12px !important;
    }
    /* Modal text, headers, and labels in dark mode */
    div[role="dialog"] p, div[role="dialog"] span, div[role="dialog"] label, div[role="dialog"] h1, div[role="dialog"] h2, div[role="dialog"] h3, div[role="dialog"] h4, div[role="dialog"] h5, div[role="dialog"] h6,
    [data-testid="stModal"] p, [data-testid="stModal"] span, [data-testid="stModal"] label, [data-testid="stModal"] h1, [data-testid="stModal"] h2, [data-testid="stModal"] h3, [data-testid="stModal"] h4, [data-testid="stModal"] h5, [data-testid="stModal"] h6 {
        color: #FFFFFF !important;
    }

    /* Checkbox overrides globally in dark mode to ensure high visibility checkmarks (excluding toggles) */
    [data-testid="stCheckbox"] [data-baseweb="checkbox"] > div:first-of-type,
    div[role="dialog"] [data-baseweb="checkbox"] > div:first-of-type,
    [data-testid="stModal"] [data-baseweb="checkbox"] > div:first-of-type {
        border: 2px solid #555555 !important;
        background-color: #2D2D2D !important;
        border-radius: 4px !important;
    }
    [data-testid="stCheckbox"] [data-baseweb="checkbox"]:has(input:checked) > div:first-of-type,
    div[role="dialog"] [data-baseweb="checkbox"]:has(input:checked) > div:first-of-type,
    [data-testid="stModal"] [data-baseweb="checkbox"]:has(input:checked) > div:first-of-type {
        background-color: #ff4b4b !important;
        border-color: #ff4b4b !important;
    }
    [data-testid="stCheckbox"] [data-baseweb="checkbox"] svg,
    div[role="dialog"] [data-baseweb="checkbox"] svg,
    [data-testid="stModal"] [data-baseweb="checkbox"] svg {
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
        color: #FFFFFF !important;
    }
    div[role="dialog"] select, [data-testid="stModal"] select {
        color: #FFFFFF !important;
        background-color: #2D2D2D !important;
        border: 1px solid #444444 !important;
    }
    div[role="dialog"] option, [data-testid="stModal"] option {
        color: #FFFFFF !important;
        background-color: #121212 !important;
    }
    div[role="dialog"] button, [data-testid="stModal"] button {
        background-color: #2D2D2D !important;
        color: #FFFFFF !important;
        border: 1px solid #444444 !important;
    }
    div[role="dialog"] button:hover, [data-testid="stModal"] button:hover {
        border-color: #ff4b4b !important;
        color: #ff4b4b !important;
        box-shadow: 0 0 8px rgba(255, 75, 75, 0.25) !important;
    }

    /* Sidebar "New Chat" button dark mode custom styling */
    div[data-testid="stSidebar"] button[key="new_chat_btn"] {
        background-color: #ff4b4b !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 10px rgba(255, 75, 75, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stSidebar"] button[key="new_chat_btn"]:hover {
        background-color: #e03e3e !important;
        box-shadow: 0 6px 15px rgba(255, 75, 75, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* Center the guest button container in the DOM */
    div.stTabs ~ div.element-container .stButton {
        display: flex !important;
        justify-content: center !important;
    }
    
    /* Guest button premium dark mode cyan/teal style */
    div.stTabs ~ div.element-container button {
        max-width: 260px !important;
        width: 260px !important;
        margin: 0 auto !important;
        background-color: #00c2cb !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 20px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 15px rgba(0, 194, 203, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    div.stTabs ~ div.element-container button:hover {
        background-color: #00a4ad !important;
        box-shadow: 0 6px 20px rgba(0, 194, 203, 0.6) !important;
        transform: translateY(-2px) !important;
    }

    /* Center the Login and Register buttons in the tabs container */
    div.stTabs .stButton {
        display: flex !important;
        justify-content: center !important;
    }

    /* Style Login and Register buttons inside tabs with Red theme */
    div.stTabs .stButton button {
        max-width: 260px !important;
        width: 260px !important;
        margin: 0 auto !important;
        background-color: #ff4b4b !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 20px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    div.stTabs .stButton button:hover {
        background-color: #e03e3e !important;
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.6) !important;
        transform: translateY(-2px) !important;
    }

    /* Compact theme & lang buttons in sidebar */
    [data-testid="stSidebar"] .settings-bar-container button {
        max-width: 100px !important;
        width: 100px !important;
        margin: 0 auto !important;
    }
    [data-testid="stSidebar"] .settings-bar-container .stButton {
        display: flex !important;
        justify-content: center !important;
    }
    """
else:
    theme_css += """
    [data-testid="stAppViewContainer"] { background-color: #FFFFFF !important; color: #1C1C1E !important; }
    [data-testid="stSidebar"] { background-color: #F8F9FA !important; color: #1C1C1E !important; border-right: 1px solid #E5E5E5; }
    [data-testid="stHeader"] { background-color: #FFFFFF !important; color: #1C1C1E !important; }
    
    /* Ensure high contrast for all markdown texts and widget labels in light mode */
    .stMarkdown, p, span, label, [data-testid="stWidgetLabel"] p, [data-testid="stExpander"] summary span { color: #1C1C1E !important; }
    
    /* Strong high contrast overrides for all titles and headers in light mode settings and panel */
    [data-testid="stAppViewContainer"] h1, 
    [data-testid="stAppViewContainer"] h2, 
    [data-testid="stAppViewContainer"] h3, 
    [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] h5,
    [data-testid="stAppViewContainer"] h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMarkdownContainer"] h6,
    .stHeading h1,
    .stHeading h2,
    .stHeading h3,
    .stHeading h4,
    .stHeading h5,
    .stHeading h6 {
        color: #1C1C1E !important;
    }
    
    /* Fix inputs, selectboxes, textareas and dropdowns in light mode (excl. slider tracks) */
    input, select, textarea, [data-baseweb="select"], [data-baseweb="select"] *, [data-baseweb="popover"] *, [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #1C1C1E !important;
    }
    
    /* Ensure high contrast for slider labels and values in light mode */
    [data-testid="stSlider"] label p, [data-testid="stSlider"] span, [data-testid="stSlider"] [data-testid="stWidgetLabel"] p {
        color: #1C1C1E !important;
    }
    /* Enforce cursor/caret visibility in light mode input fields */
    input, textarea {
        caret-color: #1C1C1E !important;
    }
    /* Form inputs and text fields in light mode */
    .stTextInput input, .stNumberInput input {
        background-color: #FFFFFF !important;
        color: #1C1C1E !important;
        border: 1px solid #CCCCCC !important;
    }
    /* Dropdown menus (select options popups) in light mode */
    div[role="listbox"], div[role="listbox"] *, ul[data-testid="main-menu-list"], ul[data-testid="main-menu-list"] * {
        background-color: #FFFFFF !important;
        color: #1C1C1E !important;
    }
    
    /* Light Mode Settings Modal - Solid Card with Transparent Backdrop Overlay */
    div[data-baseweb="modal"], div[data-baseweb="modal"] > div:first-child {
        background-color: transparent !important;
        background: transparent !important;
        backdrop-filter: none !important;
    }
    div[role="dialog"], [data-testid="stModal"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E5E5 !important;
        box-shadow: 0px 8px 32px rgba(0, 0, 0, 0.15) !important;
        border-radius: 12px !important;
    }
    /* Modal text, headers, and labels in light mode */
    div[role="dialog"] p, div[role="dialog"] span, div[role="dialog"] label, div[role="dialog"] h1, div[role="dialog"] h2, div[role="dialog"] h3, div[role="dialog"] h4, div[role="dialog"] h5, div[role="dialog"] h6,
    [data-testid="stModal"] p, [data-testid="stModal"] span, [data-testid="stModal"] label, [data-testid="stModal"] h1, [data-testid="stModal"] h2, [data-testid="stModal"] h3, [data-testid="stModal"] h4, [data-testid="stModal"] h5, [data-testid="stModal"] h6 {
        color: #1C1C1E !important;
    }
    /* Fallback color for any native Streamlit modal text elements (like "Settings" header) to ensure high contrast in light mode */
    div[role="dialog"] *, [data-testid="stModal"] * {
        color: #1C1C1E;
    }

    /* Checkbox overrides globally in light mode to ensure high visibility checkmarks (excluding toggles) */
    [data-testid="stCheckbox"] [data-baseweb="checkbox"] > div:first-of-type,
    div[role="dialog"] [data-baseweb="checkbox"] > div:first-of-type,
    [data-testid="stModal"] [data-baseweb="checkbox"] > div:first-of-type {
        border: 2px solid #CCCCCC !important;
        background-color: #FFFFFF !important;
        border-radius: 4px !important;
    }
    [data-testid="stCheckbox"] [data-baseweb="checkbox"]:has(input:checked) > div:first-of-type,
    div[role="dialog"] [data-baseweb="checkbox"]:has(input:checked) > div:first-of-type,
    [data-testid="stModal"] [data-baseweb="checkbox"]:has(input:checked) > div:first-of-type {
        background-color: #ff4b4b !important;
        border-color: #ff4b4b !important;
    }
    [data-testid="stCheckbox"] [data-baseweb="checkbox"] svg,
    div[role="dialog"] [data-baseweb="checkbox"] svg,
    [data-testid="stModal"] [data-baseweb="checkbox"] svg {
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
        color: #FFFFFF !important;
    }
    div[role="dialog"] select, [data-testid="stModal"] select {
        color: #1C1C1E !important;
        background-color: #FFFFFF !important;
        border: 1px solid #CCCCCC !important;
    }
    div[role="dialog"] option, [data-testid="stModal"] option {
        color: #1C1C1E !important;
        background-color: #FFFFFF !important;
    }
    div[role="dialog"] button, [data-testid="stModal"] button {
        background-color: #FFFFFF !important;
        color: #1C1C1E !important;
        border: 1px solid #CCCCCC !important;
    }
    div[role="dialog"] button:hover, [data-testid="stModal"] button:hover {
        border-color: #ff4b4b !important;
        color: #ff4b4b !important;
        box-shadow: 0 0 8px rgba(255, 75, 75, 0.25) !important;
    }
    
    /* Fix buttons in light mode */
    .stButton>button, [data-testid="stFormSubmitButton"]>button { background-color: #FFFFFF !important; color: #1C1C1E !important; border: 1px solid #CCC !important; }
    .stButton>button:hover, [data-testid="stFormSubmitButton"]>button:hover {
        border-color: #ff4b4b !important;
        color: #ff4b4b !important;
        box-shadow: 0 0 8px rgba(255, 75, 75, 0.25) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Fix file uploader */
    [data-testid="stFileUploaderDropzone"] { background-color: #F8F9FA !important; border: 1px dashed #CCC !important; }
    [data-testid="stFileUploaderDropzone"] * { color: #1C1C1E !important; }
    [data-testid="stFileUploaderDropzone"] button { background-color: #FFFFFF !important; color: #1C1C1E !important; border: 1px solid #999 !important; }
    [data-testid="stFileUploader"] * { color: #1C1C1E !important; }
    
    /* Fix toggle background */
    /* Fix toggle background & active state in light mode */
    [data-testid="stToggle"] [data-baseweb="checkbox"] > div:first-of-type {
        background-color: #CCCCCC !important;
        border: 1px solid #B3B3B3 !important;
    }
    [data-testid="stToggle"] [data-baseweb="checkbox"]:has(input:checked) > div:first-of-type {
        background-color: #ff4b4b !important;
        border-color: #ff4b4b !important;
    }
    [data-testid="stToggle"] [data-baseweb="checkbox"]:has(input:checked) > div:first-of-type > div {
        background-color: #FFFFFF !important;
    }
    [data-testid="stToggle"] [data-baseweb="checkbox"] p,
    [data-testid="stToggle"] [data-baseweb="checkbox"] span {
        color: #1C1C1E !important;
    }
    
    /* Fix top right menu (3 dots) text color */
    ul[data-testid="main-menu-list"] span, ul[data-testid="main-menu-list"] p, ul[data-testid="main-menu-list"] a { color: #FFFFFF !important; font-weight: 500 !important; }
    
    /* Fix chat input area in light mode */
    [data-testid="stBottom"], [data-testid="stBottom"] > div { background-color: #FFFFFF !important; }
    [data-testid="stBottomBlockContainer"] { background-color: #FFFFFF !important; }
    
    .stChatInputContainer { background-color: transparent !important; border: none !important; }
    
    [data-testid="stChatInput"] { 
        background-color: #FFFFFF !important; 
        border: 1px solid #CCCCCC !important; 
        border-radius: 24px !important; 
        overflow: hidden !important;
    }
    
    [data-testid="stChatInput"] * { 
        background-color: transparent !important; 
        color: #000000 !important; 
        caret-color: #000000 !important;
        border: none !important;
    }
    
    [data-testid="stChatInput"] textarea::placeholder { color: #666666 !important; }
    [data-testid="stChatInput"] textarea::placeholder { color: #666666 !important; }
    [data-testid="stChatInput"] button { color: #1C1C1E !important; }
    
    /* Fix status widget in light mode */
    [data-testid="stStatusWidget"] { background-color: transparent !important; border: none !important; }
    [data-testid="stStatusWidget"] * { background-color: transparent !important; color: #1C1C1E !important; }
    
    .stChatMessage { 
        background-color: #F8F9FA !important; 
        border: 1px solid #cccccc !important; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.06) !important;
    }
    
    /* Sidebar adjustments for light mode */
    [data-testid="stSidebar"] h2 { color: #4A5568 !important; }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        color: #4A5568 !important;
        background-color: transparent !important;
        border: 1px solid transparent !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background-color: rgba(255, 75, 75, 0.05) !important;
        color: #ff4b4b !important;
        border: 1px solid #ff4b4b !important;
        box-shadow: 0 0 8px rgba(255, 75, 75, 0.25) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: rgba(0, 0, 0, 0.07) !important;
        color: #1A202C !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background-color: rgba(255, 75, 75, 0.08) !important;
        color: #ff4b4b !important;
        border: 1px solid #ff4b4b !important;
        box-shadow: 0 0 8px rgba(255, 75, 75, 0.25) !important;
    }
    [data-testid="stSidebar"] [data-testid="column"] .stButton > button {
        color: #E53E3E !important;
        opacity: 0.7;
    }
    [data-testid="stSidebar"] [data-testid="column"] .stButton > button:hover {
        opacity: 1 !important;
        background-color: rgba(229, 62, 62, 0.1) !important;
        color: #C53030 !important;
    }
    
    /* Settings buttons light mode styling */
    .settings-bar-container button {
        border: 1px solid rgba(0, 0, 0, 0.12) !important;
        background-color: rgba(0, 0, 0, 0.02) !important;
        color: #4A5568 !important;
        transition: all 0.2s ease-in-out !important;
    }
    /* Settings buttons red border hover in light mode (sibling columns selector) */
    div.settings-bar-container + div .stButton button:hover,
    div.element-container:has(div.settings-bar-container) + div.element-container .stButton button:hover {
        border-color: #ff4b4b !important;
        color: #ff4b4b !important;
        box-shadow: 0 0 8px rgba(255, 75, 75, 0.25) !important;
        transform: translateY(-1px) !important;
        background-color: rgba(255, 75, 75, 0.05) !important;
    }

    /* Sidebar "New Chat" button light mode custom styling */
    div[data-testid="stSidebar"] button[key="new_chat_btn"] {
        background-color: #ff4b4b !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 10px rgba(255, 75, 75, 0.2) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stSidebar"] button[key="new_chat_btn"]:hover {
        background-color: #e03e3e !important;
        box-shadow: 0 6px 15px rgba(255, 75, 75, 0.3) !important;
        transform: translateY(-1px) !important;
    }

    /* Center the guest button container in the DOM */
    div.stTabs ~ div.element-container .stButton {
        display: flex !important;
        justify-content: center !important;
    }
    
    /* Guest button premium light mode cyan/teal style */
    div.stTabs ~ div.element-container button {
        max-width: 260px !important;
        width: 260px !important;
        margin: 0 auto !important;
        background-color: #00c2cb !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 20px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 15px rgba(0, 194, 203, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    div.stTabs ~ div.element-container button:hover {
        background-color: #00a4ad !important;
        box-shadow: 0 6px 20px rgba(0, 194, 203, 0.5) !important;
        transform: translateY(-2px) !important;
    }

    /* Center the Login and Register buttons in the tabs container */
    div.stTabs .stButton {
        display: flex !important;
        justify-content: center !important;
    }

    /* Style Login and Register buttons inside tabs with Red theme */
    div.stTabs .stButton button {
        max-width: 260px !important;
        width: 260px !important;
        margin: 0 auto !important;
        background-color: #ff4b4b !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 20px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    div.stTabs .stButton button:hover {
        background-color: #e03e3e !important;
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.5) !important;
        transform: translateY(-2px) !important;
    }

    /* Compact theme & lang buttons in sidebar */
    [data-testid="stSidebar"] .settings-bar-container button {
        max-width: 100px !important;
        width: 100px !important;
        margin: 0 auto !important;
    }
    [data-testid="stSidebar"] .settings-bar-container .stButton {
        display: flex !important;
        justify-content: center !important;
    }
    """

theme_css += "</style>"
st.markdown(theme_css, unsafe_allow_html=True)

# Initialize session state for auth
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
if "guest_mode" not in st.session_state:
    st.session_state.guest_mode = False
if "guest_id" not in st.session_state:
    st.session_state.guest_id = None

def load_users():
    if not os.path.exists("data/users.json"):
        admin_pass = os.getenv("ADMIN_PASSWORD", "admin123")
        user_pass = os.getenv("USER_PASSWORD", "user123")
        default_users = {
            "admin": {"password": admin_pass, "role": "admin"},
            "user": {"password": user_pass, "role": "user"}
        }
        os.makedirs("data", exist_ok=True)
        with open("data/users.json", "w", encoding="utf-8") as f:
            json.dump(default_users, f)
        return default_users
    else:
        with open("data/users.json", "r", encoding="utf-8") as f:
            return json.load(f)

def save_users(users):
    os.makedirs("data", exist_ok=True)
    with open("data/users.json", "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=4)

def login_page():
    lang = st.session_state.lang
    bg_b64 = get_base64_of_bin_file("login_bg.png")
    
    # Glassmorphism styling based on selected theme
    overlay_color = "rgba(10, 15, 30, 0.4)" if st.session_state.dark_theme else "rgba(240, 245, 255, 0.25)"
    text_color = "#FFFFFF" if st.session_state.dark_theme else "#1C1C1E"
    tab_bg = "rgba(20, 25, 40, 0.65)" if st.session_state.dark_theme else "rgba(255, 255, 255, 0.75)"
    tab_border = "rgba(255, 255, 255, 0.1)" if st.session_state.dark_theme else "rgba(255, 255, 255, 0.3)"
    shadow = "0 8px 32px 0 rgba(0, 0, 0, 0.5)" if st.session_state.dark_theme else "0 8px 32px 0 rgba(31, 38, 135, 0.1)"

    login_css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{bg_b64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: {overlay_color};
        z-index: 0;
    }}
    [data-testid="stAppViewContainer"] > div:first-child {{
        position: relative;
        z-index: 1;
    }}
    /* Title and Header adjustment */
    h1 {{
        color: {text_color} !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        font-weight: 800 !important;
        text-align: center;
        margin-top: 1.5rem !important;
    }}
    /* Tabs & Card styling */
    .stTabs {{
        background: {tab_bg} !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
        border-radius: 16px !important;
        border: 1px solid {tab_border} !important;
        padding: 2.5rem !important;
        box-shadow: {shadow} !important;
        max-width: 550px !important;
        margin: 2rem auto 0 auto !important;
    }}
    
    /* Input adjustments inside glassmorphism card */
    .stTabs label p {{
        color: {text_color} !important;
        font-weight: 600 !important;
    }}
    
    /* Make button stand out */
    .stButton>button {{
        border-radius: 10px !important;
        font-weight: bold !important;
        height: 3rem !important;
        transition: all 0.3s ease !important;
    }}
    
    /* Adjust tabs active states */
    button[data-baseweb="tab"] {{
        color: {text_color} !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
    }}
    button[data-baseweb="tab"]:hover {{
        color: #ff4b4b !important;
    }}
    </style>
    """
    st.markdown(login_css, unsafe_allow_html=True)

    st.title(get_text(lang, "login_title"))
    
    st.markdown("<div class='settings-bar-container'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([6, 2, 2])
    with c2:
        theme_btn_text = "☀️ Light" if st.session_state.dark_theme else "🌙 Dark"
        if st.button(theme_btn_text, use_container_width=True, key="login_theme_btn"):
            st.session_state.dark_theme = not st.session_state.dark_theme
            st.rerun()
    with c3:
        lang_btn_text = "🇬🇧 EN" if lang == "tr" else "🇹🇷 TR"
        if st.button(lang_btn_text, use_container_width=True, key="login_lang_btn"):
            st.session_state.lang = "en" if lang == "tr" else "tr"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs([get_text(lang, "login_tab"), get_text(lang, "register_tab")])
    
    users = load_users()

    with tab1:
        st.write(get_text(lang, "login_prompt"))
        username = st.text_input(get_text(lang, "username"), key="login_user")
        password = st.text_input(get_text(lang, "password"), type="password", key="login_pass")
        if st.button(get_text(lang, "login_btn"), use_container_width=True):
            if username in users and users[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.role = users[username]["role"]
                st.session_state.username = username
                st.session_state.guest_mode = False
                st.rerun()
            else:
                st.error(get_text(lang, "invalid_login"))

    with tab2:
        st.write(get_text(lang, "register_prompt"))
        new_username = st.text_input(get_text(lang, "username"), key="reg_user")
        new_password = st.text_input(get_text(lang, "password"), type="password", key="reg_pass")
        if st.button(get_text(lang, "register_btn"), use_container_width=True):
            if not new_username or not new_password:
                st.error(get_text(lang, "fill_fields"))
            elif new_username in users:
                st.warning(f"'{new_username}' {get_text(lang, 'user_exists')}")
            else:
                users[new_username] = {"password": new_password, "role": "user"}
                save_users(users)
                st.success(get_text(lang, "register_success"))

    st.markdown("<div class='guest-btn-container'>", unsafe_allow_html=True)
    if st.button(get_text(lang, "continue_as_guest"), use_container_width=True, key="guest_btn", type="secondary"):
        st.session_state.guest_mode = True
        st.session_state.guest_id = f"guest_{uuid.uuid4().hex[:8]}"
        st.session_state.username = st.session_state.guest_id
        st.session_state.logged_in = False
        st.session_state.role = "user"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def user_page():
    lang = st.session_state.lang
    st.title(get_text(lang, "app_title"))
    st.markdown(get_text(lang, "app_subtitle"))

    with st.sidebar:
        # Theme and Lang Controls
        st.markdown("<div class='settings-bar-container'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            theme_btn_text = "☀️ Light" if st.session_state.dark_theme else "🌙 Dark"
            if st.button(theme_btn_text, use_container_width=True, key="user_theme_btn"):
                st.session_state.dark_theme = not st.session_state.dark_theme
                st.rerun()
        with c2:
            lang_btn_text = "🇬🇧 EN" if lang == "tr" else "🇹🇷 TR"
            if st.button(lang_btn_text, use_container_width=True, key="user_lang_btn"):
                st.session_state.lang = "en" if lang == "tr" else "tr"
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
                
        st.divider()

        if st.button(get_text(lang, "new_chat"), use_container_width=True, type="primary", key="new_chat_btn"):
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.rerun()
            
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.header(get_text(lang, "chat_history"))
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        sessions = hm.get_user_sessions(st.session_state.get("username", "user"))
        for s in sessions:
            col1, col2 = st.columns([5, 1])
            with col1:
                title_disp = s['title']
                if len(title_disp) > 25: title_disp = title_disp[:22] + "..."
                # Active session uses primary styling, others use secondary
                is_active = st.session_state.get("current_session_id") == s["id"]
                btn_type = "primary" if is_active else "secondary"
                if st.button(title_disp, key=f"sel_{s['id']}", use_container_width=True, type=btn_type):
                    st.session_state.current_session_id = s["id"]
                    st.session_state.messages = hm.get_session_messages(s["id"])
                    st.rerun()
            with col2:
                # Math close sign (x) instead of clunky trash emoji
                if st.button("×", key=f"del_{s['id']}", use_container_width=True):
                    hm.delete_session(s["id"])
                    if st.session_state.get("current_session_id") == s["id"]:
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                    st.rerun()
                    
        st.divider()
        if not st.session_state.get("guest_mode", False):
            with st.expander(get_text(lang, "change_pass_title")):
                with st.form("user_change_pass"):
                    new_pass = st.text_input(get_text(lang, "new_password"), type="password")
                    if st.form_submit_button(get_text(lang, "change_btn")):
                        if new_pass:
                            users = load_users()
                            uname = st.session_state.username
                            if uname in users:
                                users[uname]["password"] = new_pass
                                save_users(users)
                                st.success(get_text(lang, "pass_changed_success"))
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.get("guest_mode", False):
            st.info(get_text(lang, "guest_warning"))
            if st.button(get_text(lang, "login_or_register"), use_container_width=True, type="primary"):
                st.session_state.guest_mode = False
                st.session_state.guest_id = None
                st.session_state.username = None
                st.session_state.role = None
                st.session_state.messages = []
                st.session_state.current_session_id = None
                st.rerun()
        else:
            if st.button(get_text(lang, "logout"), use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.role = None
                st.session_state.username = None
                st.session_state.messages = []
                st.session_state.current_session_id = None
                st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(get_text(lang, "chat_input_placeholder")):
        if not st.session_state.current_session_id:
            # Smart titling
            title = generate_chat_title(prompt, lang)
            st.session_state.current_session_id = hm.create_session(st.session_state.get("username", "user"), title)
            
        hm.save_message(st.session_state.current_session_id, "user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_container = st.status(get_text(lang, "status_thinking"), expanded=True)
            try:
                history = st.session_state.messages[:-1]
                
                final_response = get_text(lang, "error_msg")
                for node_name, state in run_agent_stream(prompt, history, lang):
                    if node_name == "Sorgu_Siniflandirma":
                        status_container.write(get_text(lang, "status_analyze"))
                    elif node_name == "Kural_Motoru_Node":
                        status_container.write(get_text(lang, "status_rules"))
                    elif node_name == "Vektor_Veritabani_RAG":
                        status_container.write(get_text(lang, "status_db"))
                    elif node_name == "Web_Ara_Tavily_Node":
                        status_container.write(get_text(lang, "status_web"))
                    elif node_name == "Asistan_LLM":
                        status_container.write(get_text(lang, "status_generate"))
                        final_response = state.get("yanit", get_text(lang, "error_msg"))
                    elif node_name == "Halisunasyon_Kontrol_Node":
                        status_container.write(get_text(lang, "status_reflexion"))
                
                status_container.update(label=get_text(lang, "status_complete"), state="complete", expanded=False)
                formatted_yanit = format_assistant_response(final_response)
                st.markdown(formatted_yanit)
                hm.save_message(st.session_state.current_session_id, "assistant", formatted_yanit)
                st.session_state.messages.append({"role": "assistant", "content": formatted_yanit})
                
            except Exception as e:
                status_container.update(label=get_text(lang, "status_error"), state="error", expanded=False)
                yanit = f"{get_text(lang, 'error_msg')} {str(e)}"
                st.markdown(yanit)
                hm.save_message(st.session_state.current_session_id, "assistant", yanit)
                st.session_state.messages.append({"role": "assistant", "content": yanit})

def admin_page():
    lang = st.session_state.lang
    st.markdown("<div class='settings-bar-container'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([6, 2, 2, 2])
    with col1:
        st.title("⚙️ " + ("Admin Panel" if lang == "en" else "Admin Paneli"))
    with col2:
        theme_btn_text = "☀️ Light" if st.session_state.dark_theme else "🌙 Dark"
        if st.button(theme_btn_text, use_container_width=True, key="admin_theme_btn"):
            st.session_state.dark_theme = not st.session_state.dark_theme
            st.rerun()
    with col3:
        lang_btn_text = "🇬🇧 EN" if lang == "tr" else "🇹🇷 TR"
        if st.button(lang_btn_text, use_container_width=True, key="admin_lang_btn"):
            st.session_state.lang = "en" if lang == "tr" else "tr"
            st.rerun()
    with col4:
        if st.button(get_text(lang, "logout"), use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander(get_text(lang, "change_pass_title")):
        with st.form("admin_change_pass"):
            new_pass = st.text_input(get_text(lang, "new_password"), type="password")
            if st.form_submit_button(get_text(lang, "change_btn")):
                if new_pass:
                    users = load_users()
                    uname = st.session_state.username
                    if uname in users:
                        users[uname]["password"] = new_pass
                        save_users(users)
                        st.success(get_text(lang, "pass_changed_success"))

    st.header(get_text(lang, "admin_llm_settings"))
    GROQ_MODELS = {
        "llama-3.1-8b-instant": "⚡ Llama 3.1 8B Instant",
        "llama-3.3-70b-versatile": "🦙 Llama 3.3 70B Versatile",
        "openai/gpt-oss-120b": "🤖 GPT OSS 120B",
        "openai/gpt-oss-20b": "🤖 GPT OSS 20B",
    }
    model_display = st.selectbox(
        get_text(lang, "admin_llm_model"),
        options=list(GROQ_MODELS.keys()),
        format_func=lambda x: GROQ_MODELS[x],
        index=0
    )
    os.environ["GROQ_MODEL"] = model_display

    temp = st.slider("Temperature", 0.0, 1.0, float(os.getenv("GROQ_TEMPERATURE", "0.2")), 0.05)
    os.environ["GROQ_TEMPERATURE"] = str(temp)

    rag_k = st.slider(get_text(lang, "admin_rag_chunk"), 1, 8, 4)
    os.environ["AEA_RAG_K"] = str(rag_k)

    web_search = st.toggle(get_text(lang, "admin_web_search"), value=False)
    os.environ["AEA_WEB_SEARCH"] = "1" if web_search else "0"

    st.divider()

    st.header(get_text(lang, "admin_rules_management") + " (etkilesimler.csv)")
    csv_path = "data/etkilesimler.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.dataframe(df)

        st.subheader(get_text(lang, "admin_add_interaction"))
        with st.form("add_interaction"):
            c1, c2, c3 = st.columns(3)
            with c1:
                ilac = st.text_input(get_text(lang, "admin_drug_name"))
            with c2:
                madde = st.text_input(get_text(lang, "admin_interact_substance"))
            with c3:
                risk = st.selectbox(get_text(lang, "admin_risk_level"), ["HIGH", "LOW", "NONE", "UNKNOWN"])
            
            if st.form_submit_button(get_text(lang, "admin_add_btn")):
                if ilac and madde:
                    new_row = pd.DataFrame({"ilac_adi": [ilac], "etkilesen_madde": [madde], "risk_seviyesi": [risk]})
                    df = pd.concat([df, new_row], ignore_index=True)
                    df.to_csv(csv_path, index=False)
                    st.success(get_text(lang, "admin_success_add"))
                    st.rerun()
                else:
                    st.error(get_text(lang, "admin_err_fill"))

        st.subheader(get_text(lang, "admin_remove_interaction"))
        with st.form("remove_interaction"):
            remove_idx = st.number_input(get_text(lang, "admin_remove_idx"), min_value=0, max_value=len(df)-1 if len(df)>0 else 0, step=1)
            if st.form_submit_button(get_text(lang, "admin_remove_btn")):
                if len(df) > 0 and 0 <= remove_idx < len(df):
                    df = df.drop(remove_idx)
                    df.to_csv(csv_path, index=False)
                    st.success(get_text(lang, "admin_success_del"))
                    st.rerun()
                else:
                    st.error(get_text(lang, "admin_err_idx"))
    else:
        st.warning(get_text(lang, "admin_err_csv"))

    st.divider()

    st.header(get_text(lang, "admin_pdf_management"))
    st.markdown(get_text(lang, "admin_pdf_desc"))
    
    uploaded_files = st.file_uploader(get_text(lang, "admin_pdf_select"), type="pdf", accept_multiple_files=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(get_text(lang, "admin_pdf_upload"), use_container_width=True):
            if uploaded_files:
                from pathlib import Path
                
                os.makedirs("pdfs", exist_ok=True)
                for uf in uploaded_files:
                    with open(os.path.join("pdfs", uf.name), "wb") as f:
                        f.write(uf.getbuffer())
                
                msg = get_text(lang, "status_pdf_saved").format(count=len(uploaded_files))
                st.info(msg)
                
                try:
                    from vector_db.ingest_data import ingest
                    ingest(
                        input_dir=Path("pdfs"),
                        persist_dir=Path("vector_db/chroma"),
                        collection_name=os.getenv("CHROMA_COLLECTION", "aea_kub_kt"),
                        embedding_model=os.getenv("AEA_EMBEDDING_MODEL","sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
                        glob_pattern="*.pdf",
                        chunk_size=int(os.getenv("AEA_CHUNK_SIZE", "900")),
                        chunk_overlap=int(os.getenv("AEA_CHUNK_OVERLAP", "150")),
                        clear=False
                    )
                    st.success(get_text(lang, "success_db_updated"))
                except Exception as e:
                    st.error(f"{get_text(lang, 'error_msg')} {e}")
            else:
                st.warning(get_text(lang, "warning_select_pdf"))
                
    with col_b:
        if st.button(get_text(lang, "admin_pdf_reset"), use_container_width=True):
            from pathlib import Path
            if os.path.exists("pdfs") and any(f.endswith(".pdf") for f in os.listdir("pdfs")):
                st.info(get_text(lang, "status_rebuilding_db"))
                try:
                    from vector_db.ingest_data import ingest
                    ingest(
                        input_dir=Path("pdfs"),
                        persist_dir=Path("vector_db/chroma"),
                        collection_name=os.getenv("CHROMA_COLLECTION", "aea_kub_kt"),
                        embedding_model=os.getenv("AEA_EMBEDDING_MODEL","sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
                        glob_pattern="*.pdf",
                        chunk_size=int(os.getenv("AEA_CHUNK_SIZE", "900")),
                        chunk_overlap=int(os.getenv("AEA_CHUNK_OVERLAP", "150")),
                        clear=True
                    )
                    st.success(get_text(lang, "success_db_reset"))
                except Exception as e:
                    st.error(f"{get_text(lang, 'error_msg')} {e}")
            else:
                st.warning(get_text(lang, "warning_select_pdf"))

if not st.session_state.logged_in and not st.session_state.get("guest_mode", False):
    login_page()
elif st.session_state.role == "admin":
    admin_page()
elif st.session_state.role == "user":
    user_page()
