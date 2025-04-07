"""
Configuration utilities for the interviewer module.
Loads configuration from config.yaml and overrides with environment variables from .env file.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any

# 尝试导入yaml库
try:
    import yaml
except ImportError:
    logging.warning("PyYAML package not installed. Install it with: pip install pyyaml")
    yaml = None

# 尝试导入dotenv库
try:
    from dotenv import load_dotenv
except ImportError:
    logging.warning("python-dotenv package not installed. Install it with: pip install python-dotenv")
    # Define a simple fallback if dotenv is not available
    def load_dotenv(dotenv_path=None):
        """Simple fallback for load_dotenv if python-dotenv is not installed."""
        if not dotenv_path or not os.path.exists(dotenv_path):
            return False
        with open(dotenv_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    os.environ[key] = value
                except ValueError:
                    continue
        return True

def find_file(filename: str, start_dir: str = None) -> str:
    """Find a file in the current directory or parent directories.
    
    Args:
        filename (str): Name of the file to find.
        start_dir (str, optional): Directory to start the search from.
            If None, uses the directory of this file.
            
    Returns:
        str: Path to the file if found, otherwise None.
    """
    if start_dir is None:
        start_dir = Path(__file__).parent.absolute()
    else:
        start_dir = Path(start_dir).absolute()
        
    current_dir = start_dir
    
    # Check current directory and up to 3 parent directories
    for _ in range(4):
        file_path = current_dir / filename
        if file_path.exists():
            return str(file_path)
        
        # Move up one directory
        parent_dir = current_dir.parent
        if parent_dir == current_dir:  # Reached root directory
            break
        current_dir = parent_dir
        
    return None

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    if not yaml:
        logging.error("PyYAML not installed, cannot load YAML config")
        return {}
        
    if not config_path or not os.path.exists(config_path):
        logging.warning(f"Config file not found at {config_path}")
        return {}
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading YAML config: {str(e)}")
        return {}

def load_config(config_path: str = None, env_file_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from config.yaml and override with environment variables from .env file.
    
    Args:
        config_path (str, optional): Path to the config.yaml file. If None, tries to find config.yaml in default locations.
        env_file_path (str, optional): Path to the .env file. If None, tries to find .env in default locations.
            
    Returns:
        Dict[str, Any]: Configuration dictionary for the LLM.
    """
    # Find config.yaml file if not provided
    if config_path is None:
        config_path = find_file('config.yaml')
    
    # Find .env file if not provided
    if env_file_path is None:
        env_file_path = find_file('.env')
    
    # Load .env file if it exists
    if env_file_path:
        load_dotenv(env_file_path)
        logging.info(f"Loaded environment variables from {env_file_path}")
    
    # Load config.yaml file if it exists
    yaml_config = load_yaml_config(config_path)
    
    # Extract LLM configuration from YAML
    llm_yaml_config = yaml_config.get('llm', {})  # Renamed for clarity
    provider = llm_yaml_config.get('provider', 'openai').lower()
    
    # Override with environment variables if provided
    env_provider = os.getenv('LLM_PROVIDER')
    if env_provider:
        provider = env_provider.lower()
    
    # --- Modification Starts Here ---
    final_llm_config_content = {}  # Create the content that will go inside 'llm'
    
    if provider == 'openai':
        # OpenAI configuration
        openai_config = llm_yaml_config.get('openai', {})
        models = openai_config.get('models', {})
        
        final_llm_config_content = {  # Build the content for the 'llm' key
            "provider": "openai",
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "base_url": os.getenv("OPENAI_BASE_URL", openai_config.get("base_url", "https://api.openai.com/v1")),
                # Models with environment variable overrides
                "model": os.getenv("OPENAI_DEFAULT_MODEL", models.get("default", "gpt-3.5-turbo")),
                "decision_model": os.getenv("OPENAI_DECISION_MODEL", models.get("decision", models.get("default", "gpt-3.5-turbo"))),
                "natural_question_model": os.getenv("OPENAI_NATURAL_QUESTION_MODEL", models.get("natural_question", models.get("default", "gpt-3.5-turbo"))),
                "reflection_model": os.getenv("OPENAI_REFLECTION_MODEL", models.get("reflection", models.get("default", "gpt-3.5-turbo")))
            }
        }
    elif provider == 'ollama' or provider == 'local':
        # Ollama configuration
        ollama_config = llm_yaml_config.get('ollama', {})
        models = ollama_config.get('models', {})
        
        final_llm_config_content = {  # Build the content for the 'llm' key
            "provider": "ollama",
            "ollama": {
                "base_url": os.getenv("LOCAL_LLM_BASE_URL", ollama_config.get("base_url", "http://localhost:11434")),
                # Models with environment variable overrides - corrected model access
                "models": {  # Keep the nested 'models' structure
                    "default": os.getenv("LOCAL_LLM_DEFAULT_MODEL", models.get("default", "gemma")),
                    "decision": os.getenv("LOCAL_LLM_DECISION_MODEL", models.get("decision", models.get("default", "gemma"))),
                    "natural_question": os.getenv("LOCAL_LLM_NATURAL_QUESTION_MODEL", models.get("natural_question", models.get("default", "gemma"))),
                    "reflection": os.getenv("LOCAL_LLM_REFLECTION_MODEL", models.get("reflection", models.get("default", "gemma")))
                }
            }
        }
    else:
        logging.error(f"Unsupported LLM provider: {provider}. Using OpenAI as fallback.")
        final_llm_config_content = {  # Build the content for the 'llm' key
            "provider": "openai",
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "base_url": "https://api.openai.com/v1",
                # Fallback needs a model structure too if InterviewerAgent expects it
                "model": "gpt-3.5-turbo",
                "models": {
                    "default": "gpt-3.5-turbo"
                }
            }
        }
    
    # Construct the final dictionary with the top-level 'llm' key
    final_config = {"llm": final_llm_config_content}
    # --- Modification Ends Here ---
    
    # Setup logging based on config
    if 'logging' in yaml_config:
        log_config = yaml_config['logging']
        log_level = log_config.get('level', 'INFO')
        # Avoid reconfiguring basicConfig if already configured
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=getattr(logging, log_level.upper(), logging.INFO),
                format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
    
    # Return the dictionary with the 'llm' key
    return final_config
