import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
import time
import sys
import logging
from pathlib import Path
import yaml

# Set up path for importing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities first
import app.utils as utils
from app.dataset_manager import DatasetManager

# Load configuration
config_data = utils.load_config()

# Initialize dataset manager
dataset_manager = DatasetManager(config_data)

# Import benchmark runner after initializing dataset manager
from app.benchmark_runner import BenchmarkRunner

# Set page configuration
st.set_page_config(
    page_title=config_data['application']['name'],
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'benchmark_progress' not in st.session_state:
    st.session_state.benchmark_progress = 0.0

if 'benchmark_status' not in st.session_state:
    st.session_state.benchmark_status = ""

if 'benchmark_running' not in st.session_state:
    st.session_state.benchmark_running = False

if 'benchmark_error' not in st.session_state:
    st.session_state.benchmark_error = None

# Initialize benchmark runner
benchmark_runner_instance = BenchmarkRunner()


def main():
    """
    Main application entry point
    """
    # Application title
    st.title(f"{config_data['application']['name']} v{config_data['application']['version']}")
    st.markdown("### LLM Security Benchmark Tool")
    
    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Run Benchmark", "Results", "Compare Models", "Settings"]
    )
    
    # Display appropriate page based on selection
    if page == "Home":
        home_page()
    elif page == "Run Benchmark":
        run_benchmark_page()
    elif page == "Results":
        results_page()
    elif page == "Compare Models":
        compare_models_page()
    elif page == "Settings":
        settings_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(f" 2025 {config_data['application']['name']} v{config_data['application']['version']}")


def home_page():
    """
    Home page with application overview
    """
    st.markdown("## Welcome to Purple Bench")
    st.markdown(
        """
        Purple Bench is a tool for benchmarking the security of Large Language Models (LLMs) 
        using Purple Llama's CyberSecEval 3 framework. This application helps you evaluate 
        how well different LLMs resist security threats and vulnerabilities.
        
        ### Available Benchmarks
        
        - **MITRE ATT&CK Framework**: Tests LLM resistance against tactics and techniques 
          used by adversaries to compromise systems.
        
        - **Foundational Responsible Release (FRR)**: Evaluates LLMs on cybersecurity, 
          privacy, fairness, and transparency standards.
        
        ### Getting Started
        
        1. Configure your API keys in the **Settings** page
        2. Run benchmarks against your chosen models
        3. View and compare results
        """
    )
    
    # Display quick stats
    st.markdown("## Quick Stats")
    
    col1, col2, col3 = st.columns(3)
    
    # Get benchmark results from actual result files
    benchmark_results = utils.list_benchmark_results()
    
    with col1:
        st.metric("Total Benchmarks Run", len(benchmark_results))
    
    with col2:
        # Count unique models tested
        unique_models = set()
        for result in benchmark_results:
            unique_models.add(result.get('model_name', 'Unknown'))
        st.metric("Models Tested", len(unique_models))
    
    with col3:
        # Count benchmark types
        benchmark_types = {}
        for result in benchmark_results:
            benchmark_type = result.get('benchmark_type', 'Unknown')
            benchmark_types[benchmark_type] = benchmark_types.get(benchmark_type, 0) + 1
        st.metric("Benchmark Types Used", len(benchmark_types))
    
    # Recent results
    if benchmark_results:
        st.markdown("## Recent Benchmark Results")
        
        # Convert to DataFrame for display
        df = pd.DataFrame(benchmark_results[:5])  # Show only the 5 most recent
        
        # Format timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].apply(lambda x: 
                datetime.fromisoformat(x).strftime("%Y-%m-%d %H:%M") if x else "N/A")
        
        # Select and rename columns for display
        display_cols = {
            'model_name': 'Model',
            'benchmark_type': 'Benchmark',
            'timestamp': 'Date/Time'
        }
        
        # Filter and rename columns that exist in the DataFrame
        cols_to_display = [col for col in display_cols.keys() if col in df.columns]
        df_display = df[cols_to_display].rename(columns={col: display_cols[col] for col in cols_to_display})
        
        st.dataframe(df_display, use_container_width=True)
        
        st.markdown("[View all results →](#results)")
    else:
        st.info("No benchmark results available yet. Run your first benchmark to get started!")


def run_benchmark_page():
    """
    Page for running benchmarks against LLMs
    """
    st.markdown("## Run Security Benchmark")
    
    # Check if any API keys are configured
    providers = ["openai", "anthropic", "meta", "google", "custom"]
    api_keys_available = False
    
    # Check if any API keys are available
    for provider in providers:
        if utils.get_api_key(provider):
            api_keys_available = True
            break
    
    if not api_keys_available:
        st.warning("No API keys configured. Please add API keys in the Settings page.")
        if st.button("Go to Settings"):
            st.session_state.page = "Settings"
            st.experimental_rerun()
        return
    
    # Benchmark selection
    benchmark_options = [
        {"value": "mitre", "name": "MITRE ATT&CK Framework"},
        {"value": "frr", "name": "Foundational Responsible Release (FRR)"},
        {"value": "prompt-injection", "name": "Textual Prompt Injection"},
        {"value": "visual-prompt-injection", "name": "Visual Prompt Injection"}
    ]
    
    selected_benchmark = st.selectbox(
        "Select Benchmark",
        options=[b["name"] for b in benchmark_options],
        index=0
    )
    
    # Get the selected benchmark value
    selected_benchmark_value = next((b["value"] for b in benchmark_options if b["name"] == selected_benchmark), None)
    
    # Model selection based on benchmark type
    if selected_benchmark_value == "mitre":
        st.markdown("### MITRE Benchmark Model Selection")
        st.info("The MITRE ATT&CK benchmark requires three different models for complete evaluation:")
        
        with st.expander("Why Three Models?", expanded=True):
            st.markdown("""
            The MITRE benchmark uses a three-step process:
            1. **Processing prompts** - The primary LLM generates responses to security-related prompts
            2. **Response expansion** - The expansion LLM elaborates on the initial responses
            3. **Judging expanded responses** - The judge LLM evaluates whether the expanded responses could aid in implementing a cyberattack
            """)
        
        # Primary Model Direct Input
        st.subheader("1. Primary Model (for processing prompts)")
        st.markdown("Enter the model name directly (e.g., 'gpt-4o-mini' or 'OPENAI::gpt-4o-mini')")
        primary_model_name = st.text_input(
            "Primary Model Name", 
            value="gpt-4o",
            key="primary_model_name"
        )
        primary_model_provider = st.selectbox(
            "Primary Model Provider",
            options=["openai", "anthropic", "meta", "google", "custom"],
            key="primary_provider"
        )
        
        # Create model info for the primary model
        selected_primary_info = {
            "name": primary_model_name,
            "provider": primary_model_provider
        }
        
        # Expansion Model Direct Input
        st.subheader("2. Expansion Model (for expanding responses)")
        st.markdown("Enter the model name directly (e.g., 'claude-3-sonnet' or 'ANTHROPIC::claude-3-opus')")
        expansion_model_name = st.text_input(
            "Expansion Model Name", 
            value="gpt-4o",
            key="expansion_model_name"
        )
        expansion_model_provider = st.selectbox(
            "Expansion Model Provider",
            options=["openai", "anthropic", "meta", "google", "custom"],
            key="expansion_provider"
        )
        
        # Create model info for the expansion model
        selected_expansion_info = {
            "name": expansion_model_name,
            "provider": expansion_model_provider
        }
        
        # Judge Model Direct Input
        st.subheader("3. Judge Model (for evaluating responses)")
        st.markdown("Enter the model name directly (e.g., 'claude-3-opus' or 'META::llama-3-70b')")
        judge_model_name = st.text_input(
            "Judge Model Name", 
            value="gpt-4o",
            key="judge_model_name"
        )
        judge_model_provider = st.selectbox(
            "Judge Model Provider",
            options=["openai", "anthropic", "meta", "google", "custom"],
            key="judge_provider"
        )
        
        # Create model info for the judge model
        selected_judge_info = {
            "name": judge_model_name,
            "provider": judge_model_provider
        }
            
    elif selected_benchmark_value == "prompt-injection":
        st.markdown("### Textual Prompt Injection Benchmark")  
        st.info("The Textual Prompt Injection benchmark tests an LLM's resistance to prompt injection attacks.")
        
        with st.expander("About Prompt Injection", expanded=True):
            st.markdown("""
            Textual prompt injection attacks attempt to manipulate an LLM by providing instructions that 
            override or bypass the system prompt. This benchmark tests:
            1. **Processing prompts** - The model receives prompts with injection attempts
            2. **Processing responses** - A judge model evaluates if the injection was successful
            """)
        
        # Model selection
        st.subheader("1. Model Under Test")
        st.markdown("Enter the model name to evaluate for prompt injection resistance.")
        model_name = st.text_input(
            "Model Name",
            value="gpt-4o",
            key="prompt_injection_model_name"
        )
        model_provider = st.selectbox(
            "Model Provider",
            options=["openai", "anthropic", "meta", "google", "custom"],
            key="prompt_injection_provider"
        )
        
        # Judge model selection
        st.subheader("2. Judge Model")
        st.markdown("Select a model to judge if prompt injections were successful.")
        judge_model_name = st.text_input(
            "Judge Model Name",
            value="gpt-4o",
            key="prompt_injection_judge_model_name"
        )
        judge_model_provider = st.selectbox(
            "Judge Model Provider",
            options=["openai", "anthropic", "meta", "google", "custom"],
            key="prompt_injection_judge_provider"
        )
        
        # Dataset selection
        st.subheader("3. Dataset Selection")
        try:
            # Get available datasets using dataset manager
            dataset_options = dataset_manager.get_available_datasets("prompt_injection")
            if not dataset_options:
                # Use default from config if no datasets available yet
                dataset_options = utils.load_config().get('benchmarks', {}).get('prompt_injection', {}).get('datasets', [])
                st.info("Datasets will be automatically downloaded when the benchmark runs.")
            
            # Ensure we have at least one option
            if not dataset_options:
                # Fetch from known datasets in DatasetManager
                dataset_options = dataset_manager.KNOWN_DATASETS.get("prompt_injection", [])
                
            # Last resort fallback
            if not dataset_options:
                dataset_options = ["prompt_injection.json", "prompt_injection_multilingual_machine_translated.json"]
            
            # Remove 'manual' if it exists since we now have specific datasets
            if "manual" in dataset_options:
                dataset_options.remove("manual")
                
            # Debug information
            logger = logging.getLogger('purple_bench')
            logger.info(f"Available textual prompt injection datasets: {dataset_options}")
        except Exception as e:
            logger = logging.getLogger('purple_bench')
            logger.error(f"Error fetching datasets: {str(e)}")
            # Fallback to config if there's an error
            dataset_options = utils.load_config().get('benchmarks', {}).get('prompt_injection', {}).get('datasets', [])
            if not dataset_options:
                dataset_options = dataset_manager.KNOWN_DATASETS.get("prompt_injection", [])
            if not dataset_options:
                dataset_options = ["prompt_injection.json", "prompt_injection_multilingual_machine_translated.json"]
            st.warning(f"Couldn't fetch available datasets: {str(e)}. Using default configuration.")
            
        selected_dataset = st.selectbox(
            "Select Dataset",
            options=dataset_options,
            index=0,
            key="prompt_injection_dataset"
        )
        
        # Create model info
        selected_model_info = {
            "name": model_name,
            "provider": model_provider
        }
        
        selected_judge_info = {
            "name": judge_model_name,
            "provider": judge_model_provider
        }
        
    elif selected_benchmark_value == "visual-prompt-injection":
        st.markdown("### Visual Prompt Injection Benchmark")  
        st.info("The Visual Prompt Injection benchmark tests an LLM's resistance to image-based prompt injection attacks.")
        
        with st.expander("About Visual Prompt Injection", expanded=True):
            st.markdown("""
            Visual prompt injection attacks use images containing text to inject instructions into multimodal LLMs. This benchmark tests:
            1. **Processing prompts** - The model receives prompts with image-based injection attempts
            2. **Judging responses** - A judge model evaluates if the injection was successful
            
            Multiple datasets are available, from typographic transformations to manually created test cases.
            """)
        
        # Model selection
        st.subheader("1. Model Under Test")
        st.markdown("Enter the multimodal model name to evaluate for visual prompt injection resistance.")
        model_name = st.text_input(
            "Model Name",
            value="gpt-4o",
            key="visual_prompt_injection_model_name"
        )
        model_provider = st.selectbox(
            "Model Provider",
            options=["openai", "anthropic", "meta", "google", "custom"],
            key="visual_prompt_injection_provider"
        )
        
        # Judge model selection
        st.subheader("2. Judge Model")
        st.markdown("Select a model to judge if visual prompt injections were successful.")
        judge_model_name = st.text_input(
            "Judge Model Name",
            value="gpt-4o",
            key="visual_prompt_injection_judge_model_name"
        )
        judge_model_provider = st.selectbox(
            "Judge Model Provider",
            options=["openai", "anthropic", "meta", "google", "custom"],
            key="visual_prompt_injection_judge_provider"
        )
        
        # Dataset selection
        st.subheader("3. Dataset Selection")
        try:
            # Get available datasets using dataset manager
            dataset_options = dataset_manager.get_available_datasets("visual_prompt_injection")
            
            # Ensure we have the predefined datasets even if local datasets aren't available
            if not dataset_options:
                dataset_options = dataset_manager.KNOWN_DATASETS.get("visual_prompt_injection", [])
                st.info("Datasets will be automatically downloaded when the benchmark runs.")
                
            # Always include manual option
            if "manual" not in dataset_options:
                dataset_options.append("manual")
                
            # Debug information
            logger = logging.getLogger('purple_bench')
            logger.info(f"Available visual prompt injection datasets: {dataset_options}")
        except Exception as e:
            logger = logging.getLogger('purple_bench')
            logger.error(f"Error fetching datasets: {str(e)}")
            # Fallback to config if there's an error
            dataset_options = utils.load_config().get('benchmarks', {}).get('visual_prompt_injection', {}).get('datasets', [])
            st.warning(f"Couldn't fetch available datasets: {str(e)}. Using default configuration.")
            
        selected_dataset = st.selectbox(
            "Select Dataset",
            options=dataset_options,
            index=0 if dataset_options else 0,
            key="visual_prompt_injection_dataset"
        )
        
        # Create model info
        selected_model_info = {
            "name": model_name,
            "provider": model_provider
        }
        
        selected_judge_info = {
            "name": judge_model_name,
            "provider": judge_model_provider
        }
        
    else:  # FRR benchmark only needs one model
        st.markdown("### Model Selection")
        st.markdown("Enter the model name directly (e.g., 'gpt-4o-mini' or 'ANTHROPIC::claude-3-haiku')")
        
        # Direct model input
        model_name = st.text_input(
            "Model Name",
            value="gpt-4o"
        )
        model_provider = st.selectbox(
            "Model Provider",
            options=["openai", "anthropic", "meta", "google", "custom"]
        )
        
        # Create model info
        selected_model_info = {
            "name": model_name,
            "provider": model_provider
        }
    
    # Benchmark options
    with st.expander("Benchmark Options"):
        st.markdown("Additional options for the benchmark run.")
        
        # Number of test cases (placeholder for actual options)
        num_test_cases = st.slider("Number of Test Cases", min_value=10, max_value=100, value=50, step=10)
        
        # Timeout setting
        timeout_seconds = st.number_input("Timeout (seconds)", min_value=30, max_value=300, value=60, step=10)
        
        # Parallel processing option
        parallel_processing = st.checkbox("Enable Parallel Processing", value=True)
    
    # Run benchmark button
    if st.button("Run Benchmark", use_container_width=True, disabled=st.session_state.benchmark_running):
        try:
            st.session_state.benchmark_running = True
            st.session_state.benchmark_progress = 0.0
            st.session_state.benchmark_status = "Initializing..."
            st.session_state.benchmark_error = None
            
            # Extract model information
            if selected_benchmark_value == "mitre":
                # For MITRE, we need three model configurations
                selected_primary_info = {
                    "name": primary_model_name,
                    "provider": primary_model_provider
                }
                
                selected_expansion_info = {
                    "name": expansion_model_name,
                    "provider": expansion_model_provider
                }
                
                selected_judge_info = {
                    "name": judge_model_name,
                    "provider": judge_model_provider
                }
            elif selected_benchmark_value in ["prompt-injection", "visual-prompt-injection"]:
                # For prompt injection benchmarks, we need the model under test and a judge model
                selected_model_info = {
                    "name": model_name,
                    "provider": model_provider
                }
                
                selected_judge_info = {
                    "name": judge_model_name,
                    "provider": judge_model_provider
                }
            else:
                # For other benchmarks like FRR, we need a single model configuration
                selected_model_info = {
                    "name": model_name,
                    "provider": model_provider
                }
            
            # Create progress display
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Initializing benchmark...")
            
            # Create benchmark runner
            benchmark_runner_instance = BenchmarkRunner()
            
            # Define progress callback
            def update_progress(progress: float, status: str):
                progress_bar.progress(progress)
                status_text.text(status)
                st.session_state.benchmark_progress = progress
                st.session_state.benchmark_status = status
                # Add useful debug output when verifying connections
                if "Verifying" in status or "connection" in status.lower():
                    st.info(f"🔄 {status}")
                elif "verified" in status.lower() or "successful" in status.lower():
                    st.success(f"✅ {status}")
            
            # Run the benchmark based on type
            if selected_benchmark_value == "mitre":
                # Get API keys for all three models
                primary_api_key = utils.get_api_key(selected_primary_info["provider"])
                expansion_api_key = utils.get_api_key(selected_expansion_info["provider"])
                judge_api_key = utils.get_api_key(selected_judge_info["provider"])
                
                # Validate API keys
                if not all([primary_api_key, expansion_api_key, judge_api_key]):
                    missing_providers = []
                    if not primary_api_key:
                        missing_providers.append(selected_primary_info["provider"])
                    if not expansion_api_key:
                        missing_providers.append(selected_expansion_info["provider"])
                    if not judge_api_key:
                        missing_providers.append(selected_judge_info["provider"])
                    
                    st.error(f"API keys missing for: {', '.join(missing_providers)}. Please configure them in Settings.")
                    st.session_state.benchmark_running = False
                    return
                
                # Show models being used
                with st.expander("Models Being Tested", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"**Primary Model:** {selected_primary_info['name']} ({selected_primary_info['provider']})")
                    with col2:
                        st.info(f"**Expansion Model:** {selected_expansion_info['name']} ({selected_expansion_info['provider']})")
                    with col3:
                        st.info(f"**Judge Model:** {selected_judge_info['name']} ({selected_judge_info['provider']})")
                
                # Run MITRE benchmark with all three models
                results = benchmark_runner_instance.run_mitre_benchmark(
                    primary_model={
                        "name": selected_primary_info["name"],
                        "provider": selected_primary_info["provider"],
                        "api_key": primary_api_key
                    },
                    expansion_model={
                        "name": selected_expansion_info["name"],
                        "provider": selected_expansion_info["provider"],
                        "api_key": expansion_api_key
                    },
                    judge_model={
                        "name": selected_judge_info["name"],
                        "provider": selected_judge_info["provider"],
                        "api_key": judge_api_key
                    },
                    num_test_cases=num_test_cases,
                    timeout=timeout_seconds,
                    parallel=parallel_processing,
                    callback=update_progress
                )
            elif selected_benchmark_value in ["prompt-injection", "visual-prompt-injection"]:
                # Get the selected dataset if applicable
                dataset = selected_dataset if "selected_dataset" in locals() else None
                
                # For prompt injection benchmarks, we need the model under test and a judge model
                api_key = utils.get_api_key(selected_model_info["provider"])
                
                # Update status for clarity
                status_text.text(f"Running {selected_benchmark_value} benchmark with {selected_model_info['name']}...")
                
                # Ensure we're using the correct dataset variable
                if selected_benchmark_value == 'visual-prompt-injection':
                    # Run the benchmark with the dataset parameter
                    status_text.text(f"Running {selected_benchmark_value} benchmark with {selected_model_info['name']} using dataset: {dataset}...")
                    
                    results = benchmark_runner_instance.run_benchmark(
                        model_name=selected_model_info['name'],
                        provider=selected_model_info['provider'],
                        api_key=api_key,
                        benchmark_type=selected_benchmark_value,
                        dataset=dataset,  # Pass the selected dataset
                        callback=update_progress
                    )
                else:
                    results = benchmark_runner_instance.run_benchmark(
                        model_name=selected_model_info['name'],
                        provider=selected_model_info['provider'],
                        api_key=api_key,
                        benchmark_type=selected_benchmark_value,
                        callback=update_progress
                    )
                
            else:
                # For other benchmarks like FRR, use the general method with one model
                api_key = utils.get_api_key(selected_model_info["provider"])
                
                # Update status for clarity
                status_text.text(f"Running {selected_benchmark_value} benchmark with {selected_model_info['name']}...")
                
                # Run the benchmark
                results = benchmark_runner_instance.run_benchmark(
                    selected_model_info["name"],
                    selected_model_info["provider"],
                    api_key,
                    selected_benchmark_value,
                    callback=update_progress
                )
            
            # Check for errors
            if "error" in results:
                st.error(f"Error running benchmark: {results['error']}")
                st.session_state.benchmark_error = results['error']
                st.session_state.benchmark_running = False
                return
                
            # Display detailed debug information if available
            if "debug_info" in results:
                with st.expander("Debug Information", expanded=False):
                    st.code(results["debug_info"])
            
            # Show completion message
            st.success("Benchmark completed successfully!")
            
            # Display results
            display_benchmark_results(results)
            
        except Exception as e:
            st.error(f"Error running benchmark: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")
            st.session_state.benchmark_error = str(e)
        finally:
            st.session_state.benchmark_running = False
    
    # Display progress if benchmark is running
    if st.session_state.benchmark_running:
        st.progress(st.session_state.benchmark_progress)
        st.text(st.session_state.benchmark_status)


def results_page():
    """
    Page for viewing benchmark results
    """
    st.markdown("## Benchmark Results")
    
    # Get all benchmark results using the utility function
    results = utils.list_benchmark_results()
    
    if not results:
        st.info("No benchmark results available yet. Run a benchmark to see results here.")
        return
    
    # Initialize session state for delete confirmation if it doesn't exist
    if 'delete_confirmation' not in st.session_state:
        st.session_state.delete_confirmation = None
    if 'deleted_file' not in st.session_state:
        st.session_state.deleted_file = None
        
    # Display confirmation message if a file was deleted
    if st.session_state.deleted_file:
        st.success(f"Benchmark result deleted successfully.")
        st.session_state.deleted_file = None
    
    # Handle delete confirmation
    if st.session_state.delete_confirmation:
        filepath = st.session_state.delete_confirmation
        col1, col2 = st.columns([3, 1])
        with col1:
            st.warning(f"Are you sure you want to delete this benchmark result? This action cannot be undone.")
        with col2:
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("Yes, Delete"):
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        st.session_state.deleted_file = filepath
                        st.session_state.delete_confirmation = None
                        st.rerun()
                    else:
                        st.error("Failed to delete benchmark result.")
            with cancel_col:
                if st.button("Cancel"):
                    st.session_state.delete_confirmation = None
                    st.rerun()
    
    # Convert to DataFrame for filtering and display
    df = pd.DataFrame(results)
    
    # Format timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'].apply(lambda x: 
            datetime.fromisoformat(x).strftime("%Y-%m-%d %H:%M") if x else "N/A")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        # Model filter
        if 'model_name' in df.columns:
            all_models = sorted(df['model_name'].unique())
            selected_models = st.multiselect(
                "Filter by Model",
                options=all_models,
                default=[]
            )
    
    with col2:
        # Benchmark type filter
        if 'benchmark_type' in df.columns:
            all_benchmarks = sorted(df['benchmark_type'].unique())
            selected_benchmarks = st.multiselect(
                "Filter by Benchmark Type",
                options=all_benchmarks,
                default=[]
            )
    
    # Apply filters
    filtered_df = df.copy()
    if 'model_name' in df.columns and selected_models:
        filtered_df = filtered_df[filtered_df['model_name'].isin(selected_models)]
    
    if 'benchmark_type' in df.columns and selected_benchmarks:
        filtered_df = filtered_df[filtered_df['benchmark_type'].isin(selected_benchmarks)]
    
    # Select and rename columns for display
    display_cols = {
        'model_name': 'Model',
        'benchmark_type': 'Benchmark Type',
        'dataset': 'Dataset',
        'timestamp': 'Date/Time',
        'version': 'App Version'
    }
    
    # Filter and rename columns that exist in the DataFrame
    cols_to_display = [col for col in display_cols.keys() if col in filtered_df.columns]
    df_display = filtered_df[cols_to_display].rename(columns={col: display_cols[col] for col in cols_to_display})
    
    # Add invisible column for action buttons
    df_display['Actions'] = ""
    
    # Display results table
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # Create a button for each row
    st.write("Select a benchmark result to view or delete:")
    for i, row in filtered_df.iterrows():
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            model_name = row.get('model_name', 'Unknown')
            benchmark_type = row.get('benchmark_type', 'Unknown')
            dataset = row.get('dataset', '')
            timestamp = row.get('timestamp', 'Unknown')
            
            benchmark_info = f"**{model_name}** - {benchmark_type}"
            if dataset and benchmark_type == 'visual-prompt-injection':
                benchmark_info += f" - Dataset: {dataset}"
            benchmark_info += f" - {timestamp}"
            
            st.write(benchmark_info)
        with col2:
            if st.button("View", key=f"view_{i}"):
                st.session_state.selected_result = row['filepath']
                st.rerun()
        with col3:
            if st.button("Delete", key=f"delete_{i}", type="primary", use_container_width=True):
                st.session_state.delete_confirmation = row['filepath']
                st.rerun()
        st.markdown("---")
    
    # Result details
    st.markdown("### Result Details")
    
    # Select result to view
    if 'filename' in filtered_df.columns:
        selected_result = st.selectbox(
            "Select Result to View",
            options=filtered_df['filename'].tolist(),
            format_func=lambda x: f"{filtered_df[filtered_df['filename'] == x]['model_name'].iloc[0]} - "
                               f"{filtered_df[filtered_df['filename'] == x]['benchmark_type'].iloc[0]} - "
                               f"{filtered_df[filtered_df['filename'] == x]['timestamp'].iloc[0]}"
        )
        
        if selected_result:
            # Get filepath for the selected result
            filepath = filtered_df[filtered_df['filename'] == selected_result]['filepath'].iloc[0]
            
            # Load the result file
            try:
                # Load result data from the selected file rather than from config
                result = utils.load_benchmark_results(filepath)
                
                # Display result details
                st.markdown(f"#### Benchmark Results for {result.get('metadata', {}).get('model_name', 'Model')}")
                
                # Overall score
                st.metric("Overall Score", f"{result.get('overall_score', 0.0):.2f}")
                
                # Category scores
                st.markdown("##### Category Scores")
                
                # Convert category scores to DataFrame for visualization
                categories = result.get('categories', {})
                category_data = {
                    'Category': [],
                    'Score': [],
                    'Tests Passed': [],
                    'Tests Total': []
                }
                
                for category, data in categories.items():
                    category_data['Category'].append(category.capitalize())
                    category_data['Score'].append(data.get('score', 0.0))
                    category_data['Tests Passed'].append(data.get('tests_passed', 0))
                    category_data['Tests Total'].append(data.get('tests_total', 0))
                
                df_categories = pd.DataFrame(category_data)
                
                # Create bar chart
                chart = alt.Chart(df_categories).mark_bar().encode(
                    x=alt.X('Category:N', title='Category'),
                    y=alt.Y('Score:Q', title='Score', scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color('Category:N', legend=None),
                    tooltip=['Category', 'Score', 'Tests Passed', 'Tests Total']
                ).properties(
                    width=600,
                    height=300
                )
                
                st.altair_chart(chart, use_container_width=True)
                
                # Detailed test results
                st.markdown("##### Detailed Test Results")
                
                detailed_results = result.get('detailed_results', [])
                if detailed_results:
                    # Convert to DataFrame
                    df_tests = pd.DataFrame(detailed_results)
                    
                    # Add pass/fail emoji
                    if 'passed' in df_tests.columns:
                        df_tests['Status'] = df_tests['passed'].apply(lambda x: " Pass" if x else " Fail")
                    
                    # Select columns to display
                    display_cols = ['test_id', 'name', 'category', 'Status', 'details']
                    display_cols = [col for col in display_cols if col in df_tests.columns]
                    
                    # Rename columns
                    rename_map = {
                        'test_id': 'Test ID',
                        'name': 'Test Name',
                        'category': 'Category',
                        'details': 'Details'
                    }
                    
                    df_display = df_tests[display_cols].rename(columns={col: rename_map.get(col, col) for col in display_cols})
                    
                    # Display table
                    st.dataframe(df_display, use_container_width=True)
                else:
                    st.info("No detailed test results available.")
                
            except Exception as e:
                st.error(f"Error loading result file: {str(e)}")


def compare_models_page():
    """
    Page for comparing benchmark results from multiple models
    """
    st.markdown("## Compare Benchmark Results")
    
    # Get all benchmark results from files
    benchmark_results = utils.list_benchmark_results()
    
    if not benchmark_results:
        st.info("No benchmark results available yet. Run benchmarks to compare results here.")
        return
    
    # Convert to DataFrame for filtering and selection
    df = pd.DataFrame(benchmark_results)
    
    # Format timestamp if needed
    if 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'].apply(lambda x: 
            datetime.fromisoformat(x).strftime("%Y-%m-%d %H:%M") if x else "N/A")
    
    # Select results to compare (up to 4)
    st.markdown("### Select Results to Compare (up to 4)")
    
    if 'filepath' in df.columns and 'model_name' in df.columns and 'benchmark_type' in df.columns and 'timestamp' in df.columns:
        # Create selection options
        options = [f"{row['model_name']} - {row['benchmark_type']} - {row['timestamp']}" for _, row in df.iterrows()]
        values = df['filepath'].tolist()
        
        selected_results = st.multiselect(
            "Select Results",
            options=options,
            format_func=lambda x: x,
            max_selections=4
        )
        
        # Map selected options back to filepaths
        selected_filepaths = [values[options.index(result)] for result in selected_results if result in options]
        
        if len(selected_filepaths) > 1:
            # Compare the selected results
            comparison = benchmark_runner_instance.compare_results(selected_filepaths)
            
            if 'error' in comparison:
                st.error(comparison['error'])
            else:
                # Display comparison
                st.markdown("### Comparison Results")
                
                # Overall scores comparison
                st.markdown("#### Overall Scores")
                
                # Create DataFrame for overall scores
                overall_scores = comparison.get('overall_scores', {})
                overall_data = {
                    'Model': list(overall_scores.keys()),
                    'Score': list(overall_scores.values())
                }
                
                df_overall = pd.DataFrame(overall_data)
                
                # Create bar chart for overall scores
                chart = alt.Chart(df_overall).mark_bar().encode(
                    x=alt.X('Model:N', title='Model'),
                    y=alt.Y('Score:Q', title='Score', scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color('Model:N', legend=None),
                    tooltip=['Model', 'Score']
                ).properties(
                    width=600,
                    height=300
                )
                
                st.altair_chart(chart, use_container_width=True)
                
                # Category scores comparison
                st.markdown("#### Category Scores")
                
                # Create DataFrame for category scores
                category_scores = comparison.get('category_scores', {})
                category_data = []
                
                for category, scores in category_scores.items():
                    for model, score in scores.items():
                        category_data.append({
                            'Category': category.capitalize(),
                            'Model': model,
                            'Score': score
                        })
                
                df_categories = pd.DataFrame(category_data)
                
                # Create grouped bar chart for category scores
                if not df_categories.empty:
                    chart = alt.Chart(df_categories).mark_bar().encode(
                        x=alt.X('Category:N', title='Category'),
                        y=alt.Y('Score:Q', title='Score', scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color('Model:N', title='Model'),
                        column=alt.Column('Model:N', title=None),
                        tooltip=['Category', 'Model', 'Score']
                    ).properties(
                        width=150,
                        height=300
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No category scores available for comparison.")
                
                # Test results comparison
                st.markdown("#### Test Results Comparison")
                
                # Create DataFrame for test results
                test_results = comparison.get('test_results', {})
                test_data = []
                
                for test_id, test_info in test_results.items():
                    for model, result in test_info.get('results', {}).items():
                        test_data.append({
                            'Test ID': test_id,
                            'Test Name': test_info.get('name', ''),
                            'Category': test_info.get('category', '').capitalize(),
                            'Model': model,
                            'Passed': result.get('passed', False),
                            'Details': result.get('details', '')
                        })
                
                df_tests = pd.DataFrame(test_data)
                
                if not df_tests.empty:
                    # Add pass/fail emoji
                    df_tests['Status'] = df_tests['Passed'].apply(lambda x: " Pass" if x else " Fail")
                    
                    # Create pivot table for test results
                    pivot_df = df_tests.pivot_table(
                        index=['Test ID', 'Test Name', 'Category'],
                        columns='Model',
                        values='Status',
                        aggfunc='first'
                    ).reset_index()
                    
                    # Display table
                    st.dataframe(pivot_df, use_container_width=True)
                else:
                    st.info("No test results available for comparison.")
        elif len(selected_filepaths) == 1:
            st.warning("Please select at least 2 results to compare (up to 4).")


def settings_page():
    """
    Settings page for API key management and application configuration
    """
    st.markdown("## Settings")
    
    # Create tabs for different settings sections
    tabs = st.tabs(["API Keys", "Application Settings", "About"])
    
    with tabs[0]:
        # API key management
        st.markdown("### API Key Management")
        st.info("Configure API keys for different model providers.")
        
        # Load current config
        config_data = utils.load_config()
        
        # Get available providers
        providers = ["openai", "anthropic", "meta", "google", "custom"]
        
        # Display API key input for each provider
        for provider in providers:
            st.subheader(provider.capitalize())
            api_key = st.text_input(
                f"{provider.capitalize()} API Key",
                value=utils.get_api_key(provider) or '',
                type="password"
            )
            
            # Save API key button
            if st.button(f"Save {provider.capitalize()} API Key"):
                # Save the API key using the utility function
                if utils.save_api_key(provider, api_key, save_locally=True):
                    st.success(f"{provider.capitalize()} API key saved successfully!")
                else:
                    st.error(f"Failed to save {provider.capitalize()} API key.")
    
    with tabs[1]:
        # Application settings
        st.markdown("### Application Settings")
        
        # Load current config
        config_data = utils.load_config()
        
        # UI theme
        theme = st.selectbox(
            "UI Theme",
            options=["light", "dark"],
            index=0 if config_data.get('ui', {}).get('theme', 'light') == "light" else 1
        )
        
        # Results directory
        results_dir = st.text_input(
            "Results Directory",
            value=config_data.get('application', {}).get('results_directory', '')
        )
        
        # Log level
        log_level = st.selectbox(
            "Log Level",
            options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(
                config_data.get('application', {}).get('log_level', 'INFO'))
        )
        
        # Save settings button
        if st.button("Save Settings"):
            # Update settings in config file
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.yaml')
            
            try:
                # Ensure we have the proper structure
                if 'ui' not in config_data:
                    config_data['ui'] = {}
                if 'application' not in config_data:
                    config_data['application'] = {}
                
                # Update values
                config_data['ui']['theme'] = theme
                config_data['application']['results_directory'] = results_dir
                config_data['application']['log_level'] = log_level
                
                # Save to file
                with open(config_path, 'w') as file:
                    yaml.dump(config_data, file, default_flow_style=False)
                
                st.success("Settings saved successfully!")
                
                # Apply log level change immediately
                logging.getLogger('purple_bench').setLevel(getattr(logging, log_level))
            except Exception as e:
                st.error(f"Failed to save settings: {str(e)}")
    
    with tabs[2]:
        # About section
        st.markdown("### About Purple Bench")
        st.markdown(
            """
            Purple Bench is a tool for benchmarking the security of Large Language Models (LLMs) 
            using Purple Llama's CyberSecEval 3 framework.
            
            **GitHub Repository**: [https://github.com/yourusername/purple-bench](https://github.com/yourusername/purple-bench)
            
            **License**: MIT
            """
        )


def display_benchmark_results(results):
    """
    Display benchmark results
    """
    st.markdown("### Results Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Overall Score", f"{results.get('overall_score', 0.0):.2f}")
    
    with col2:
        # Count passed tests
        passed_tests = sum(1 for test in results.get('detailed_results', []) if test.get('passed', False))
        total_tests = len(results.get('detailed_results', []))
        st.metric("Tests Passed", f"{passed_tests}/{total_tests}")
    
    # Category scores
    st.markdown("#### Category Scores")
    
    # Convert category scores to DataFrame for visualization
    categories = results.get('categories', {})
    category_data = {
        'Category': [],
        'Score': []
    }
    
    for category, data in categories.items():
        category_data['Category'].append(category.capitalize())
        category_data['Score'].append(data.get('score', 0.0))
    
    df_categories = pd.DataFrame(category_data)
    
    # Create bar chart
    chart = alt.Chart(df_categories).mark_bar().encode(
        x=alt.X('Category:N', title='Category'),
        y=alt.Y('Score:Q', title='Score', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Category:N', legend=None),
        tooltip=['Category', 'Score']
    ).properties(
        width=600,
        height=300
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # Link to detailed results
    st.markdown(f"[View detailed results →](#results)")


if __name__ == "__main__":
    main()
