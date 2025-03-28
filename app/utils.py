import os
import yaml
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', f'purple_bench_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('purple_bench')


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the configuration file. If None, uses default path.
        
    Returns:
        Dict containing configuration
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
        raise


def save_api_key(service_name: str, api_key: str, save_locally: bool = False) -> bool:
    """
    Save API key securely
    
    Args:
        service_name: Name of the service (e.g., 'openai', 'anthropic')
        api_key: The API key to save
        save_locally: Whether to save the key to disk (otherwise just in session)
        
    Returns:
        bool: Success status
    """
    try:
        import streamlit as st
        
        # Always save to session state
        if 'api_keys' not in st.session_state:
            st.session_state.api_keys = {}
        
        st.session_state.api_keys[service_name] = api_key
        
        # Optionally save to disk
        if save_locally:
            api_keys_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
            api_keys_file = os.path.join(api_keys_dir, 'api_keys.yaml')
            
            # Load existing keys if file exists
            existing_keys = {}
            if os.path.exists(api_keys_file):
                with open(api_keys_file, 'r') as file:
                    existing_keys = yaml.safe_load(file) or {}
            
            # Update with new key
            existing_keys[service_name] = api_key
            
            # Save back to file
            with open(api_keys_file, 'w') as file:
                yaml.dump(existing_keys, file)
            
            logger.info(f"API key for {service_name} saved locally")
        
        return True
    except Exception as e:
        logger.error(f"Failed to save API key for {service_name}: {str(e)}")
        return False


def get_api_key(service_name: str) -> Optional[str]:
    """
    Retrieve API key
    
    Args:
        service_name: Name of the service to get key for
        
    Returns:
        str: API key if found, None otherwise
    """
    try:
        import streamlit as st
        
        # First check session state
        if 'api_keys' in st.session_state and service_name in st.session_state.api_keys:
            return st.session_state.api_keys[service_name]
        
        # Then check local file
        api_keys_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'api_keys.yaml')
        if os.path.exists(api_keys_file):
            with open(api_keys_file, 'r') as file:
                keys = yaml.safe_load(file) or {}
                if service_name in keys:
                    # Save to session state for future use
                    if 'api_keys' not in st.session_state:
                        st.session_state.api_keys = {}
                    st.session_state.api_keys[service_name] = keys[service_name]
                    return keys[service_name]
        
        return None
    except Exception as e:
        logger.error(f"Failed to retrieve API key for {service_name}: {str(e)}")
        return None


def generate_result_filename(model_name: str, benchmark_type: str, dataset: Optional[str] = None) -> str:
    """
    Generate a unique filename for benchmark results
    
    Args:
        model_name: Name of the model being benchmarked
        benchmark_type: Type of benchmark (e.g., 'mitre', 'frr')
        dataset: Optional dataset name used for the benchmark
        
    Returns:
        str: Unique filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace('/', '_').replace(' ', '_')
    dataset_suffix = f"_{dataset.replace('.', '_')}" if dataset else ""
    return f"{safe_model_name}_{benchmark_type}{dataset_suffix}_{timestamp}.json"


def save_benchmark_results(results: Dict[str, Any], model_name: str, benchmark_type: str, dataset: Optional[str] = None) -> str:
    """
    Save benchmark results to file
    
    Args:
        results: Benchmark results to save
        model_name: Name of the model
        benchmark_type: Type of benchmark
        dataset: Optional dataset name used for the benchmark
        
    Returns:
        str: Path to saved file
    """
    try:
        config = load_config()
        results_dir = Path(config['application']['results_directory'])
        
        # Create directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename and path
        filename = generate_result_filename(model_name, benchmark_type, dataset)
        filepath = results_dir / filename
        
        # Add metadata
        results['metadata'] = {
            'model_name': model_name,
            'benchmark_type': benchmark_type,
            'timestamp': datetime.now().isoformat(),
            'version': config['application']['version']
        }
        
        # Add dataset info if provided
        if dataset:
            results['metadata']['dataset'] = dataset
        
        # Save to file
        with open(filepath, 'w') as file:
            json.dump(results, file, indent=2)
        
        logger.info(f"Benchmark results saved to {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Failed to save benchmark results: {str(e)}")
        raise


def load_benchmark_results(filepath: str) -> Dict[str, Any]:
    """
    Load benchmark results from file
    
    Args:
        filepath: Path to results file
        
    Returns:
        Dict: Benchmark results
    """
    try:
        with open(filepath, 'r') as file:
            results = json.load(file)
        return results
    except Exception as e:
        logger.error(f"Failed to load benchmark results from {filepath}: {str(e)}")
        raise


def list_benchmark_results() -> List[Dict[str, Any]]:
    """
    List all available benchmark results
    
    Returns:
        List of dicts with result file information
    """
    try:
        config = load_config()
        results_dir = Path(config['application']['results_directory'])
        
        # Create directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        results = []
        for file in results_dir.glob('*.json'):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    metadata = data.get('metadata', {})
                    
                    results.append({
                        'filename': file.name,
                        'filepath': str(file),
                        'model_name': metadata.get('model_name', 'Unknown'),
                        'benchmark_type': metadata.get('benchmark_type', 'Unknown'),
                        'timestamp': metadata.get('timestamp', ''),
                        'version': metadata.get('version', ''),
                        'dataset': metadata.get('dataset', 'Unknown'),
                    })
            except Exception as e:
                logger.warning(f"Could not process result file {file}: {str(e)}")
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return results
    except Exception as e:
        logger.error(f"Failed to list benchmark results: {str(e)}")
        return []