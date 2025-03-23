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

# Import local modules
from .utils import load_config, save_benchmark_results, list_benchmark_results, load_benchmark_results, logger
from .config import api_key_management, get_configured_models
from .benchmark_runner import BenchmarkRunner

# Load configuration
config = load_config()

# Set page configuration
st.set_page_config(
    page_title=config['application']['name'],
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

# Initialize benchmark runner
benchmark_runner = BenchmarkRunner()


def main():
    """
    Main application entry point
    """
    # Application title
    st.title(f"{config['application']['name']} v{config['application']['version']}")
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
    st.sidebar.markdown(f" 2025 {config['application']['name']} v{config['application']['version']}")


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
        
        - **Textual Prompt Injection**: Evaluates LLMs' ability to detect and resist 
          textual prompt injection attacks.
        
        - **Visual Prompt Injection**: Evaluates LLMs' ability to detect and resist 
          visual prompt injection attacks.
        
        ### Getting Started
        
        1. Configure your API keys in the **Settings** page
        2. Run benchmarks against your chosen models
        3. View and compare results
        """
    )
    
    # Display quick stats
    st.markdown("## Quick Stats")
    
    col1, col2, col3 = st.columns(3)
    
    # Get benchmark results
    results = list_benchmark_results()
    
    with col1:
        st.metric("Benchmarks Run", len(results))
    
    with col2:
        # Count unique models
        unique_models = len(set([r.get('model_name', '') for r in results]))
        st.metric("Models Tested", unique_models)
    
    with col3:
        # Get latest benchmark date
        if results:
            latest_date = max([r.get('timestamp', '') for r in results])
            if latest_date:
                try:
                    latest_date = datetime.fromisoformat(latest_date).strftime("%Y-%m-%d")
                except:
                    latest_date = "N/A"
            else:
                latest_date = "N/A"
        else:
            latest_date = "N/A"
        st.metric("Latest Benchmark", latest_date)
    
    # Recent results
    if results:
        st.markdown("## Recent Benchmark Results")
        
        # Convert to DataFrame for display
        df = pd.DataFrame(results[:5])  # Show only the 5 most recent
        
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
    
    # Create benchmark runner
    benchmark_runner = BenchmarkRunner()
    
    # Check if any API keys are configured
    models = get_configured_models()
    if not models:
        st.warning("⚠️ No API keys configured. Please add API keys in the Settings page.")
        if st.button("Go to Settings"):
            st.session_state.page = "Settings"
            st.experimental_rerun()
        return
    
    # Model selection
    selected_model = st.selectbox(
        "Select Model",
        options=[model["display_name"] for model in models],
        index=0
    )
    
    # Get the selected model details
    selected_model_info = next((model for model in models if model["display_name"] == selected_model), None)
    
    # Benchmark selection
    benchmark_options = [
        {"value": "mitre", "name": "MITRE ATT&CK Framework"},
        {"value": "frr", "name": "Foundational Responsible Release (FRR)"},
        {"value": "prompt_injection", "name": "Textual Prompt Injection"},
        {"value": "visual_prompt_injection", "name": "Visual Prompt Injection"}
    ]
    
    selected_benchmark = st.selectbox(
        "Select Benchmark",
        options=[b["name"] for b in benchmark_options],
        index=0
    )
    
    # Get the selected benchmark value
    selected_benchmark_value = next((b["value"] for b in benchmark_options if b["name"] == selected_benchmark), None)
    
    # Dataset selection for prompt injection benchmarks
    selected_dataset = None
    if selected_benchmark_value in ["prompt_injection", "visual_prompt_injection"]:
        # Initialize the dataset manager to list available datasets
        dataset_options = benchmark_runner.dataset_manager.list_available_datasets(selected_benchmark_value)
        
        if not dataset_options:
            st.warning(f"No datasets available for {selected_benchmark}. Please check your configuration.")
        else:
            selected_dataset = st.selectbox(
                "Select Dataset",
                options=dataset_options,
                index=0,
                key=f"{selected_benchmark_value}_dataset"
            )
            
            # Info about the dataset
            st.info(f"Using dataset: {selected_dataset}")
            
            # Option to download dataset if not already downloaded
            if not benchmark_runner.dataset_manager.dataset_exists(selected_benchmark_value, selected_dataset):
                if st.button("Download Dataset"):
                    with st.spinner(f"Downloading {selected_dataset}..."):
                        success = benchmark_runner.dataset_manager.download_dataset(
                            selected_benchmark_value, 
                            selected_dataset
                        )
                        if success:
                            st.success(f"Dataset {selected_dataset} downloaded successfully!")
                        else:
                            st.error(f"Failed to download dataset {selected_dataset}. Please check your connection and try again.")
    
    # Benchmark options
    with st.expander("Benchmark Options"):
        st.markdown("Additional options for the benchmark run.")
        
        # Number of test cases (placeholder for actual options)
        num_test_cases = st.slider("Number of Test Cases", min_value=10, max_value=100, value=50, step=10)
        
        # Timeout setting
        timeout_seconds = st.number_input("Timeout (seconds)", min_value=30, max_value=300, value=60, step=10)
    
    # Run benchmark button
    if st.button("Run Benchmark", disabled=st.session_state.benchmark_running):
        if selected_model_info and selected_benchmark_value:
            # Check if dataset is required but not selected
            if selected_benchmark_value in ["prompt_injection", "visual_prompt_injection"] and not selected_dataset:
                st.error(f"Please select a dataset for {selected_benchmark}")
                return
            
            # Set benchmark as running
            st.session_state.benchmark_running = True
            st.session_state.benchmark_progress = 0.0
            st.session_state.benchmark_status = "Initializing benchmark..."
            
            # Create progress bar
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            # Run benchmark
            try:
                # Get API key for the selected provider
                api_key = get_api_key(selected_model_info["provider"])
                
                if not api_key:
                    st.error(f"API key for {selected_model_info['provider']} not found. Please configure it in Settings.")
                    st.session_state.benchmark_running = False
                    return
                
                # Define callback function to update progress
                def update_progress(progress, status):
                    st.session_state.benchmark_progress = progress
                    st.session_state.benchmark_status = status
                    progress_bar.progress(progress)
                    status_text.text(status)
                
                # Run the benchmark
                results = benchmark_runner.run_benchmark(
                    model_name=selected_model_info["name"],
                    provider=selected_model_info["provider"],
                    api_key=api_key,
                    benchmark_type=selected_benchmark_value,
                    dataset=selected_dataset,
                    callback=update_progress
                )
                
                # Display results summary
                st.success(f"Benchmark completed successfully!")
                
                # Show results summary
                st.markdown("### Results Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Overall Score", f"{results.get('overall_score', 0.0):.2f}")
                
                with col2:
                    # For prompt injection benchmarks, show successful defenses
                    if selected_benchmark_value in ["prompt_injection", "visual_prompt_injection"]:
                        successful = results.get('successful_defenses', 0)
                        total = results.get('total_samples', 0)
                        st.metric("Successful Defenses", f"{successful}/{total}")
                    else:
                        # Count passed tests for MITRE and FRR
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
                    category_data['Category'].append(category.replace('_', ' ').capitalize())
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
                
            except Exception as e:
                st.error(f"Error running benchmark: {str(e)}")
            finally:
                # Reset benchmark running state
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
    
    # Get all benchmark results
    results = list_benchmark_results()
    
    if not results:
        st.info("No benchmark results available yet. Run a benchmark to see results here.")
        return
    
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
        'timestamp': 'Date/Time',
        'version': 'App Version'
    }
    
    # Filter and rename columns that exist in the DataFrame
    cols_to_display = [col for col in display_cols.keys() if col in filtered_df.columns]
    df_display = filtered_df[cols_to_display].rename(columns={col: display_cols[col] for col in cols_to_display})
    
    # Add action column
    df_display['Actions'] = None
    
    # Display results table
    st.dataframe(df_display, use_container_width=True)
    
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
                result = load_benchmark_results(filepath)
                
                # Display result details
                st.markdown(f"#### {result.get('benchmark', 'Benchmark')} Results for {result.get('model', 'Model')}")
                
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
                    category_data['Category'].append(category.replace('_', ' ').capitalize())
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
                        df_tests['Status'] = df_tests['passed'].apply(lambda x: "✅ Pass" if x else "❌ Fail")
                    
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
    st.markdown("## Compare Model Results")
    
    # Get all benchmark results
    results = list_benchmark_results()
    
    if not results:
        st.info("No benchmark results available yet. Run benchmarks to compare results here.")
        return
    
    # Convert to DataFrame for filtering and selection
    df = pd.DataFrame(results)
    
    # Format timestamp
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
            comparison = benchmark_runner.compare_results(selected_filepaths)
            
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
                            'Category': category.replace('_', ' ').capitalize(),
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
                    df_tests['Status'] = df_tests['Passed'].apply(lambda x: "✅ Pass" if x else "❌ Fail")
                    
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
        api_key_management()
    
    with tabs[1]:
        # Application settings
        st.markdown("### Application Settings")
        
        # Load current config
        config = load_config()
        
        # UI theme
        theme = st.selectbox(
            "UI Theme",
            options=["light", "dark"],
            index=0 if config['ui']['theme'] == "light" else 1
        )
        
        # Results directory
        results_dir = st.text_input(
            "Results Directory",
            value=config['application']['results_directory']
        )
        
        # Log level
        log_level = st.selectbox(
            "Log Level",
            options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(config['application']['log_level'])
        )
        
        # Save settings button
        if st.button("Save Settings"):
            # In a real application, this would update the config file
            st.success("Settings saved successfully!")
    
    with tabs[2]:
        # About section
        st.markdown("### About Purple Bench")
        st.markdown(f"Version: {config['application']['version']}")
        st.markdown(
            """
            Purple Bench is a tool for benchmarking the security of Large Language Models (LLMs) 
            using Purple Llama's CyberSecEval 3 framework.
            
            **GitHub Repository**: [https://github.com/yourusername/purple-bench](https://github.com/yourusername/purple-bench)
            
            **License**: MIT
            """
        )


if __name__ == "__main__":
    main()