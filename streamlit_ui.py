from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire

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
from agent import ZaubAgent
from ai_providers import get_model_list
from data_classes import AgentDeps
from prompts import get_agent_prompt

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

# Agent will be stored in session state instead of global variable

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal['user', 'model']
    timestamp: str
    content: str

def create_sidebar():
    """Create sidebar with model selection options."""
    st.sidebar.title("Configurações do Chatbot")
    
    # Get available models
    model_list = get_model_list()
    
    # Initialize saved config in session state if not exists
    if "saved_config" not in st.session_state:
        st.session_state.saved_config = {
            "embedding_provider": list(model_list["embedding"].keys())[0],
            "embedding_model": list(model_list["embedding"][list(model_list["embedding"].keys())[0]].keys())[0],
            "language_provider": list(model_list["language"].keys())[0],
            "language_model": model_list["language"][list(model_list["language"].keys())[0]][0],
            "max_chunks": 10,
            "similarity_threshold": 0.7
        }
    
    # Current (temporary) selections
    st.sidebar.subheader("Provedor de Embeddings")
    embedding_providers = list(model_list["embedding"].keys())
    current_embedding_provider = st.sidebar.selectbox(
        "Escolha o Provedor de Embeddings",
        embedding_providers,
        index=embedding_providers.index(st.session_state.saved_config["embedding_provider"]) if st.session_state.saved_config["embedding_provider"] in embedding_providers else 0,
        key="temp_embedding_provider"
    )
    
    # Embedding Model Selection
    available_embedding_models = list(model_list["embedding"][current_embedding_provider].keys())
    current_embedding_model = st.sidebar.selectbox(
        "Escolha o Modelo de Embeddings",
        available_embedding_models,
        index=available_embedding_models.index(st.session_state.saved_config["embedding_model"]) if st.session_state.saved_config["embedding_model"] in available_embedding_models else 0,
        key="temp_embedding_model"
    )
    
    # Show embedding dimensions
    embedding_dim = model_list["embedding"][current_embedding_provider][current_embedding_model]
    st.sidebar.divider()
    
    # Language Provider Selection  
    st.sidebar.subheader("Provedor de Linguagem")
    language_providers = list(model_list["language"].keys())
    current_language_provider = st.sidebar.selectbox(
        "Escolha o Provedor de Linguagem", 
        language_providers,
        index=language_providers.index(st.session_state.saved_config["language_provider"]) if st.session_state.saved_config["language_provider"] in language_providers else 0,
        key="temp_language_provider"
    )
    
    # Language Model Selection
    available_language_models = model_list["language"][current_language_provider]
    current_language_model = st.sidebar.selectbox(
        "Escolha o Modelo de Linguagem",
        available_language_models,
        index=available_language_models.index(st.session_state.saved_config["language_model"]) if st.session_state.saved_config["language_model"] in available_language_models else 0,
        key="temp_language_model"
    )
    
    st.sidebar.divider()
    
    # Agent Configuration
    st.sidebar.subheader("Configurações do Agent")
    current_max_chunks = st.sidebar.slider("Máximo de Chunks", min_value=1, max_value=20, value=st.session_state.saved_config["max_chunks"], key="temp_max_chunks")
    current_similarity_threshold = st.sidebar.slider("Limite de similaridade", min_value=0.0, max_value=1.0, value=st.session_state.saved_config["similarity_threshold"], step=0.1, key="temp_similarity_threshold")
    
    # Create current config dict
    current_config = {
        "embedding_provider": current_embedding_provider,
        "embedding_model": current_embedding_model, 
        "language_provider": current_language_provider,
        "language_model": current_language_model,
        "max_chunks": current_max_chunks,
        "similarity_threshold": current_similarity_threshold,
        "embedding_dim": embedding_dim
    }
    
    # Check if there are unsaved changes
    config_changed = (
        current_config["embedding_provider"] != st.session_state.saved_config["embedding_provider"] or
        current_config["embedding_model"] != st.session_state.saved_config["embedding_model"] or
        current_config["language_provider"] != st.session_state.saved_config["language_provider"] or
        current_config["language_model"] != st.session_state.saved_config["language_model"] or
        current_config["max_chunks"] != st.session_state.saved_config["max_chunks"] or
        current_config["similarity_threshold"] != st.session_state.saved_config["similarity_threshold"]
    )
    
    # Save button
    st.sidebar.divider()
    if config_changed:
        st.sidebar.warning("⚠️ Alterações não salvas")
        
    if st.sidebar.button("Salvar Configurações", disabled=not config_changed):
        # Save the current config
        st.session_state.saved_config = {
            "embedding_provider": current_config["embedding_provider"],
            "embedding_model": current_config["embedding_model"],
            "language_provider": current_config["language_provider"],
            "language_model": current_config["language_model"],
            "max_chunks": current_config["max_chunks"],
            "similarity_threshold": current_config["similarity_threshold"]
        }
        # Clear agent to force recreation with new config
        st.session_state.pop("agent_instance", None)
        st.session_state.pop("agent_config", None)
        st.sidebar.success("✅ Configurações salvas!")
        st.rerun()
    
    # Return the saved config (which is what the agent should use)
    return {
        "embedding_provider": st.session_state.saved_config["embedding_provider"],
        "embedding_model": st.session_state.saved_config["embedding_model"], 
        "language_provider": st.session_state.saved_config["language_provider"],
        "language_model": st.session_state.saved_config["language_model"],
        "max_chunks": st.session_state.saved_config["max_chunks"],
        "similarity_threshold": st.session_state.saved_config["similarity_threshold"],
        "embedding_dim": model_list["embedding"][st.session_state.saved_config["embedding_provider"]][st.session_state.saved_config["embedding_model"]]
    }

async def get_or_create_agent(config):
    """Get or create agent instance based on current configuration."""
    
    # Create a config signature for comparison
    config_signature = {
        "embedding_provider": config["embedding_provider"],
        "embedding_model": config["embedding_model"],
        "language_provider": config["language_provider"],
        "language_model": config["language_model"],
        "max_chunks": config["max_chunks"],
        "similarity_threshold": config["similarity_threshold"]
    }
    
    # Check if we have an agent in session state and if config has changed
    if ("agent_instance" in st.session_state and 
        "agent_config" in st.session_state and 
        st.session_state.agent_config == config_signature):
        # Configuration hasn't changed, return existing agent
        st.sidebar.info("🔵 Reutilizando agente existente")
        return st.session_state.agent_instance
    
    # Close existing agent if it exists
    if "agent_instance" in st.session_state:
        try:
            await st.session_state.agent_instance.close()
        except:
            pass
    
    # Create new agent with dependencies
    agent_deps = AgentDeps(
        system_prompt=get_agent_prompt(),
        max_chunks=config["max_chunks"],
        similarity_threshold=config["similarity_threshold"]
    )
    
    # Create embedding model string and pydantic model string
    embedding_model_string = f"{config['embedding_provider']}:{config['embedding_model']}"
    pydantic_model_string = f"{config['language_provider']}:{config['language_model']}"
    
    agent = ZaubAgent(
        embedding_model=embedding_model_string,
        pydantic_model=pydantic_model_string,
        deps=agent_deps,
        global_deps=global_deps
    )
    
    # Initialize the agent
    try:
        st.sidebar.info("Criando novo agente...")
        await agent.initialize()
        st.sidebar.success("Novo agente inicializado com sucesso!")
        
        # Store in session state
        st.session_state.agent_instance = agent
        st.session_state.agent_config = config_signature
        
        return agent
    except Exception as e:
        st.sidebar.error(f"Erro ao iniciar o Agente: {e}")
        return None


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


async def run_agent_with_streaming(user_input: str, agent_instance, config):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    if agent_instance is None:
        st.error("Agente não inicializado corretamente. Por favor, verifique suas configurações.")
        return
    
    try:
        # Check if streaming is supported for this provider
        supports_streaming = config["language_provider"] != "anthropic"
        
        if supports_streaming:
            # Use streaming for supported providers (OpenAI, HuggingFace)
            async with agent_instance.run_stream(
                user_input,
                message_history=st.session_state.messages[:-1],  # pass entire conversation so far
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
        else:
            # Use non-streaming for Anthropic models
            with st.spinner("🤖 Processando com Claude..."):
                result = await agent_instance.run(user_input)
                
                # Display the response
                response_text = result.data
                st.markdown(response_text)
                
                # Add new messages from this run, excluding user-prompt messages
                filtered_messages = [msg for msg in result.new_messages() 
                                    if not (hasattr(msg, 'parts') and 
                                            any(part.part_kind == 'user-prompt' for part in msg.parts))]
                st.session_state.messages.extend(filtered_messages)

                # Add the final response to the messages
                st.session_state.messages.append(
                    ModelResponse(parts=[TextPart(content=response_text)])
                )
            
    except Exception as e:
        st.error(f"Erro ao executar o Agente: {e}")
        # Add error message to conversation
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=f"Desculpe, encontrei um erro: {str(e)}")])
        )


async def main():
    st.set_page_config(
        page_title="Zaub Expert", 
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Zaub Expert")
    st.write("O Oráculo para tudo que há de Zaub no time de Produto.")
    
    # Create sidebar with model selection
    config = create_sidebar()
    
    # Display current configuration in main area
    with st.expander("Configuração atual", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Embedding:**")
            st.write(f"- Provedor: `{config['embedding_provider']}`")
            st.write(f"- Modelo: `{config['embedding_model']}`")
            st.write(f"- Dimensões: `{config['embedding_dim']}`")
        with col2:
            st.write("**Linguagem:**")
            st.write(f"- Provedor: `{config['language_provider']}`")
            st.write(f"- Modelo: `{config['language_model']}`")
        # Clean up markdown-breaking characters for display
        clean_prompt = (get_agent_prompt()
                        .replace('`', '')           # Remove backticks
                        .replace('•', '-')         # Replace bullet points
                        .replace('"', '"')          # Replace smart quotes
                        .replace('"', '"')          # Replace smart quotes  
                        .replace('—', '-')          # Replace em dash
                        .replace('\n\n', '\n')      # Reduce double line breaks
                        .replace('-', ' ')          # Reduce double spaces
                        .replace('"', ' ')          # Reduce double spaces
                        .strip())
        st.write(f"- System Prompt: ```{clean_prompt[:200]}...```")


    # Get or create agent based on current configuration
    with st.spinner("Inicializando Agente..."):
        agent_instance = await get_or_create_agent(config)

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
    user_input = st.chat_input("Pergunte o que quiser sobre a Zaub")

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
            await run_agent_with_streaming(user_input, agent_instance, config)

    # Add a button to clear conversation
    if st.button("Limpar Conversa"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    asyncio.run(main())