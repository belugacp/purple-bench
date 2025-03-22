import streamlit as st
import os
from typing import Dict, Any, Optional, Tuple, List
from .utils import save_api_key, get_api_key, load_config, logger


def api_key_management() -> None:
    """
    Render the API key management section in the Streamlit UI
    """
    st.header("API Key Management")
    st.markdown(
        "Configure API keys for different LLM providers. Keys can be saved locally for convenience or entered each session."
    )
    
    # Get list of supported providers from config
    config = load_config()
    
    # Create tabs for different providers
    providers = ["OpenAI", "Anthropic", "Meta", "Google", "Custom"]
    tabs = st.tabs(providers)
    
    for i, provider in enumerate(providers):
        with tabs[i]:
            configure_provider_api(provider)


def configure_provider_api(provider: str) -> None:
    """
    Configure API for a specific provider
    
    Args:
        provider: The provider name (e.g., 'OpenAI', 'Anthropic')
    """
    # Convert provider name to lowercase for key storage
    provider_key = provider.lower()
    
    # Get existing API key if available
    existing_key = get_api_key(provider_key)
    masked_key = mask_api_key(existing_key) if existing_key else ""
    
    # Provider-specific instructions
    if provider == "OpenAI":
        st.markdown("Enter your OpenAI API key. You can find this in your [OpenAI dashboard](https://platform.openai.com/api-keys).")
    elif provider == "Anthropic":
        st.markdown("Enter your Anthropic API key. You can find this in your [Anthropic console](https://console.anthropic.com/keys).")
    elif provider == "Meta":
        st.markdown("Enter your Meta API key for accessing Llama models.")
    elif provider == "Google":
        st.markdown("Enter your Google API key for accessing Gemini models.")
    elif provider == "Custom":
        st.markdown("Enter a custom API key and provider name.")
        custom_provider = st.text_input("Custom Provider Name", key=f"custom_provider_name")
        if custom_provider:
            provider_key = custom_provider.lower()
            existing_key = get_api_key(provider_key)
            masked_key = mask_api_key(existing_key) if existing_key else ""
    
    # API Key input
    col1, col2 = st.columns([3, 1])
    with col1:
        if existing_key:
            new_key = st.text_input(
                "API Key", 
                value=masked_key,
                type="password",
                key=f"api_key_{provider_key}",
                placeholder="Enter API key"
            )
        else:
            new_key = st.text_input(
                "API Key", 
                type="password",
                key=f"api_key_{provider_key}",
                placeholder="Enter API key"
            )
    
    # Save locally checkbox
    save_locally = st.checkbox("Save API key locally", key=f"save_locally_{provider_key}")
    
    # Save button
    if st.button("Save API Key", key=f"save_button_{provider_key}"):
        # Only save if the key is not the masked version of the existing key
        if new_key and new_key != masked_key:
            if save_api_key(provider_key, new_key, save_locally):
                st.success(f"{provider} API key saved successfully!")
                # Set session state to indicate API is configured
                st.session_state[f"{provider_key}_api_configured"] = True
            else:
                st.error(f"Failed to save {provider} API key.")
    
    # Display API status
    display_api_status(provider_key)


def mask_api_key(api_key: str) -> str:
    """
    Mask API key for display
    
    Args:
        api_key: The API key to mask
        
    Returns:
        str: Masked API key
    """
    if not api_key:
        return ""
    
    # Show first 4 and last 4 characters, mask the rest
    if len(api_key) <= 8:
        return "*" * len(api_key)
    else:
        return api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]


def display_api_status(provider_key: str) -> None:
    """
    Display API connection status
    
    Args:
        provider_key: The provider key to check status for
    """
    api_key = get_api_key(provider_key)
    
    if api_key:
        # Check if we've already tested this API key in this session
        if f"{provider_key}_api_configured" in st.session_state and st.session_state[f"{provider_key}_api_configured"]:
            st.success("✅ API key configured")
        else:
            # Here we would normally test the API key, but for now just mark as configured
            st.session_state[f"{provider_key}_api_configured"] = True
            st.success("✅ API key configured")
    else:
        st.warning("⚠️ API key not configured")


def test_api_connection(provider_key: str, api_key: str) -> bool:
    """
    Test API connection to verify key works
    
    Args:
        provider_key: The provider key (e.g., 'openai', 'anthropic')
        api_key: The API key to test
        
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        # Implement provider-specific API connection tests
        if provider_key == "openai":
            import openai
            client = openai.OpenAI(api_key=api_key)
            # Simple model list call to test connection
            client.models.list()
            return True
        elif provider_key == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            # Simple models call to test connection
            client.models.list()
            return True
        # Add other providers as needed
        else:
            # For unknown providers, just assume it works
            logger.warning(f"No API test implemented for provider: {provider_key}")
            return True
    except Exception as e:
        logger.error(f"API connection test failed for {provider_key}: {str(e)}")
        return False


def get_configured_models() -> List[Dict[str, Any]]:
    """
    Get a list of configured models based on available API keys
    
    Returns:
        List of dicts with model information
    """
    models = []
    
    # Check for OpenAI API key
    if get_api_key("openai"):
        models.extend([
            {"provider": "openai", "name": "gpt-4o", "display_name": "GPT-4o (OpenAI)"},
            {"provider": "openai", "name": "gpt-4-turbo", "display_name": "GPT-4 Turbo (OpenAI)"},
            {"provider": "openai", "name": "gpt-3.5-turbo", "display_name": "GPT-3.5 Turbo (OpenAI)"}
        ])
    
    # Check for Anthropic API key
    if get_api_key("anthropic"):
        models.extend([
            {"provider": "anthropic", "name": "claude-3-opus", "display_name": "Claude 3 Opus (Anthropic)"},
            {"provider": "anthropic", "name": "claude-3-sonnet", "display_name": "Claude 3 Sonnet (Anthropic)"},
            {"provider": "anthropic", "name": "claude-3-haiku", "display_name": "Claude 3 Haiku (Anthropic)"}
        ])
    
    # Check for Meta API key
    if get_api_key("meta"):
        models.extend([
            {"provider": "meta", "name": "llama-3-70b", "display_name": "Llama 3 70B (Meta)"},
            {"provider": "meta", "name": "llama-3-8b", "display_name": "Llama 3 8B (Meta)"}
        ])
    
    # Check for Google API key
    if get_api_key("google"):
        models.extend([
            {"provider": "google", "name": "gemini-pro", "display_name": "Gemini Pro (Google)"},
            {"provider": "google", "name": "gemini-ultra", "display_name": "Gemini Ultra (Google)"}
        ])
    
    # Add any custom providers
    for key in st.session_state.keys():
        if key.endswith("_api_configured") and st.session_state[key]:
            provider = key.replace("_api_configured", "")
            if provider not in ["openai", "anthropic", "meta", "google"]:
                models.append({
                    "provider": provider,
                    "name": f"custom-{provider}",
                    "display_name": f"Custom Model ({provider.capitalize()})"
                })
    
    return models