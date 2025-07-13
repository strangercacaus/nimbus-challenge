from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from agent import cloudwalk_expert, CloudwalkDeps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)          


async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    deps = CloudwalkDeps(
        supabase=supabase,
        openai_client=openai_client
    )

    # Run the agent in a stream
    async with cloudwalk_expert.run_stream(
        user_input,
        deps=deps,
        message_history= st.session_state.messages[:-1],  # pass entire conversation so far
    ) as result:
        # We'll gather partial text to show incrementally
        partial_text = ""
        message_placeholder = st.empty()

        # Render partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Now that the stream is finished, we have a final result.
        # Add new messages from this run, excluding user-prompt messages
        filtered_messages = [msg for msg in result.new_messages() 
                            if not (hasattr(msg, 'parts') and 
                                    any(part.part_kind == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)

        # Add the final response to the messages
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )


async def main():
    # Custom CSS to force light mode and add custom colors
    st.markdown("""
    <style>
    /* Force light mode and custom color scheme */
    .stApp {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
    }
    
    /* Override Streamlit's theme toggle - force light mode */
    .stApp [data-testid="stHeader"] {
        background-color: #f8f9fa !important;
        border-bottom: 1px solid #e9ecef !important;
    }
    
    /* Hide the theme toggle button */
    .stApp [data-testid="stToolbar"] [data-testid="stHeaderActionElements"] button[title="Change theme"] {
        display: none !important;
    }
    
    /* Main container */
    .stApp [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }
    
    /* Sidebar styling */
    .stApp [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        border-right: 1px solid #e9ecef !important;
    }
    
    /* Chat message styling */
    .stApp [data-testid="stChatMessage"] {
        background-color: #f8f9fa !important;
        border: 1px solid #e9ecef !important;
        border-radius: 8px !important;
        margin-bottom: 10px !important;
    }
    
    /* User message styling */
    .stApp [data-testid="stChatMessage"][data-testid*="user"] {
        background-color: #e3f2fd !important;
        border-color: #2196f3 !important;
    }
    
    /* Assistant message styling */
    .stApp [data-testid="stChatMessage"][data-testid*="assistant"] {
        background-color: #f1f8e9 !important;
        border-color: #4caf50 !important;
    }
    
    /* Chat input styling */
    .stApp [data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border: 2px solid #2196f3 !important;
        border-radius: 8px !important;
    }
    
    .stApp [data-testid="stChatInput"] input {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
        border: none !important;
    }
    
    .stApp [data-testid="stChatInput"] button {
        background-color: #2196f3 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
    }
    
    .stApp [data-testid="stChatInput"] button:hover {
        background-color: #1976d2 !important;
    }
    
    /* Text styling */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #1976d2 !important;
        font-weight: 600 !important;
    }
    
    .stApp p, .stApp div, .stApp span {
        color: #2c3e50 !important;
    }
    
    /* Markdown content styling */
    .stApp .stMarkdown {
        color: #2c3e50 !important;
    }
    
    .stApp .stMarkdown code {
        background-color: #f5f5f5 !important;
        color: #d32f2f !important;
        padding: 2px 4px !important;
        border-radius: 4px !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    .stApp .stMarkdown pre {
        background-color: #f5f5f5 !important;
        color: #2c3e50 !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        padding: 16px !important;
    }
    
    /* Links styling */
    .stApp a {
        color: #1976d2 !important;
    }
    
    .stApp a:hover {
        color: #0d47a1 !important;
    }
    
    /* Success/info message styling */
    .stApp .stSuccess {
        background-color: #e8f5e8 !important;
        color: #2e7d32 !important;
        border: 1px solid #4caf50 !important;
    }
    
    .stApp .stInfo {
        background-color: #e3f2fd !important;
        color: #1976d2 !important;
        border: 1px solid #2196f3 !important;
    }
    
    /* Override any dark mode attempts */
    .stApp * {
        color: inherit !important;
    }
    
    /* Custom accent colors */
    :root {
        --primary-color: #2196f3;
        --secondary-color: #4caf50;
        --accent-color: #ff9800;
        --background-color: #ffffff;
        --text-color: #2c3e50;
        --light-bg: #f8f9fa;
        --border-color: #e9ecef;
    }
    
    /* Animation for smooth transitions */
    .stApp [data-testid="stChatMessage"] {
        transition: all 0.3s ease !important;
    }
    
    .stApp [data-testid="stChatMessage"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("CloudWalk Expert")
    st.write("Ask any question about CloudWalk, the hidden beauty of this ecosystem lie within.")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("Take your curiosity for a walk.")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            await run_agent_with_streaming(user_input)


if __name__ == "__main__":
    asyncio.run(main())
