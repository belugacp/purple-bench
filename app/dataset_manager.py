import os
import subprocess
import shutil
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

# Configure logging
logger = logging.getLogger('purple_bench')

class DatasetManager:
    """Manages downloading and accessing datasets for benchmarks"""
    
    # S3 bucket for Purple Llama datasets
    S3_BUCKET = "s3://purplellama-cyberseceval"
    CYBERSECEVAL_PATH = "cyberseceval3"
    
    # Dataset paths relative to S3 bucket root
    DATASET_PATHS = {
        "prompt_injection": f"{CYBERSECEVAL_PATH}/prompt_injection",
        "visual_prompt_injection": f"{CYBERSECEVAL_PATH}/visual_prompt_injection"
    }
    
    # Known datasets for each type (from S3)
    KNOWN_DATASETS = {
        "visual_prompt_injection": [
            "cse2_typographic_images",
            "cse2_visual_overlays",
            "cse2_adversarial_patches",
            "cse2_adversarial_qr_codes"
        ],
        "prompt_injection": [
            "prompt_injection.json",
            "prompt_injection_multilingual_machine_translated.json"
        ]
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the dataset manager with the application config"""
        self.config = config
        
        # Create data/datasets directory if it doesn't exist
        self.datasets_base_dir = os.path.join(
            Path(self.config['application']['results_directory']).parent, 
            "datasets"
        )
        os.makedirs(self.datasets_base_dir, exist_ok=True)
        
        # Create .gitignore in the datasets directory if it doesn't exist
        self._create_datasets_gitignore()
        
    def _create_datasets_gitignore(self):
        """Create a .gitignore file in the datasets directory to exclude everything"""
        gitignore_path = os.path.join(self.datasets_base_dir, ".gitignore")
        if not os.path.exists(gitignore_path):
            with open(gitignore_path, "w") as f:
                f.write("# Ignore all files in this directory\n")
                f.write("*\n")
                f.write("# Except this file\n")
                f.write("!.gitignore\n")
            logger.info(f"Created .gitignore in {self.datasets_base_dir}")
            
    def check_aws_cli_installed(self) -> bool:
        """Check if AWS CLI is installed"""
        try:
            subprocess.run(
                ["aws", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def download_dataset(self, dataset_type: str, dataset_name: Optional[str] = None) -> str:
        """Download a dataset from S3 if it doesn't exist locally
        
        Args:
            dataset_type: Type of dataset (e.g., 'prompt_injection', 'visual_prompt_injection')
            dataset_name: Optional specific dataset name (e.g., 'cse2_typographic_images')
            
        Returns:
            Path to the downloaded dataset
        """
        if dataset_type not in self.DATASET_PATHS:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
        # Check if AWS CLI is installed
        if not self.check_aws_cli_installed():
            logger.warning("AWS CLI not installed. Please install it with 'pip install awscli'")
            raise RuntimeError(
                "AWS CLI is required to download datasets. "
                "Please install it with 'pip install awscli'"
            )
        
        # Determine the destination path
        dest_path = os.path.join(self.datasets_base_dir, dataset_type)
        os.makedirs(dest_path, exist_ok=True)
        
        # Determine the S3 source path
        if dataset_name and dataset_type == "visual_prompt_injection":
            # For visual prompt injection, we need a specific subdirectory
            s3_path = f"{self.S3_BUCKET}/{self.DATASET_PATHS[dataset_type]}/{dataset_name}"
            local_path = os.path.join(dest_path, dataset_name)
        else:
            # For other datasets, download everything in the category
            s3_path = f"{self.S3_BUCKET}/{self.DATASET_PATHS[dataset_type]}"
            local_path = dest_path
            
        # Only download if it doesn't exist already
        if dataset_name and os.path.exists(local_path):
            logger.info(f"Dataset {dataset_name} already exists at {local_path}")
            return local_path
            
        # Download the dataset
        logger.info(f"Downloading dataset from {s3_path} to {local_path}")
        try:
            # Create the directory if it doesn't exist
            os.makedirs(local_path, exist_ok=True)
            
            # Run the AWS CLI command to download the dataset
            subprocess.run(
                ["aws", "--no-sign-request", "s3", "cp", 
                 "--recursive", s3_path, local_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            logger.info(f"Successfully downloaded dataset to {local_path}")
            return local_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download dataset: {e.stderr.decode() if e.stderr else str(e)}")
            raise RuntimeError(f"Failed to download dataset from {s3_path}")
    
    def get_dataset_path(self, dataset_type: str, dataset_name: Optional[str] = None, download: bool = True) -> str:
        """Get the path to a dataset, downloading it if necessary
        
        Args:
            dataset_type: Type of dataset (e.g., 'prompt_injection', 'visual_prompt_injection')
            dataset_name: Optional specific dataset name (e.g., 'cse2_typographic_images')
            download: Whether to download the dataset if it doesn't exist locally
            
        Returns:
            Path to the dataset
        """
        # Convert to proper format for dataset paths
        normalized_type = dataset_type.replace("-", "_")
        
        # Determine the local path
        local_base_path = os.path.join(self.datasets_base_dir, normalized_type)
        
        if dataset_name:
            local_path = os.path.join(local_base_path, dataset_name)
        else:
            local_path = local_base_path
            
        # If the dataset doesn't exist locally and download is True, download it
        if download and not os.path.exists(local_path):
            return self.download_dataset(normalized_type, dataset_name)
            
        return local_path
    
    def get_available_datasets(self, dataset_type: str) -> List[str]:
        """
        Get list of available datasets for a given type
        
        Args:
            dataset_type: Type of dataset (e.g., 'prompt_injection', 'visual_prompt_injection')
            
        Returns:
            List of dataset names
        """
        # Convert to proper format for dataset paths
        normalized_type = dataset_type.replace("-", "_")
        logger.info(f"Getting available datasets for {normalized_type}")
        
        # Start with the known predefined datasets for this type
        all_datasets = []
        if normalized_type in self.KNOWN_DATASETS:
            all_datasets.extend(self.KNOWN_DATASETS[normalized_type])
            logger.info(f"Added {len(self.KNOWN_DATASETS[normalized_type])} known datasets for {normalized_type}")
        
        # Add any local datasets that are already downloaded
        local_path = os.path.join(self.datasets_base_dir, normalized_type)
        local_datasets = []
        
        if os.path.exists(local_path):
            if normalized_type == "visual_prompt_injection":
                local_datasets = [d for d in os.listdir(local_path) 
                        if os.path.isdir(os.path.join(local_path, d)) and not d.startswith(".")]
            else:
                # For text-based datasets, get JSON files
                local_datasets = [f for f in os.listdir(local_path) 
                    if f.endswith(".json") and not f.startswith(".")]
            
            # Add local datasets that aren't already in the list
            for dataset in local_datasets:
                if dataset not in all_datasets:
                    all_datasets.append(dataset)
            
            logger.info(f"Found {len(local_datasets)} local datasets for {normalized_type}")
        
        # Add manual option if not already in the list
        if "manual" not in all_datasets:
            all_datasets.append("manual")
        
        # If fallback needed, use config
        if not all_datasets and normalized_type in self.config.get('benchmarks', {}):
            config_datasets = self.config['benchmarks'][normalized_type].get('datasets', [])
            all_datasets.extend(config_datasets)
            logger.info(f"Using {len(config_datasets)} datasets from config for {normalized_type}")
            
        logger.info(f"Final dataset list for {normalized_type}: {all_datasets}")
        return all_datasets
