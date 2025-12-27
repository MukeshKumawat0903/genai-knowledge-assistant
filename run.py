"""
GenAI Knowledge Assistant - Application Entry Point

This script runs the Streamlit application.

Usage:
    python run.py

Or with Streamlit directly:
    streamlit run run.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import after path is set
from app import __version__
from app.ui.chat_app import main
from app.utils.config import get_settings
from app.utils.logger import setup_logger


def initialize_app():
    """
    Initialize application components.
    
    TODO:
    - Load configuration
    - Set up logging
    - Initialize any required resources
    - Validate environment
    """
    # Load configuration
    config = get_settings()
    
    # Set up logging
    logger = setup_logger(
        name="genai_assistant",
        log_level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info("GenAI Knowledge Assistant Starting...")
    logger.info("=" * 60)
    logger.info(f"Version: {__version__}")
    logger.info(f"Debug Mode: {config.debug}")
    
    # TODO: Add more initialization logic
    # - Check API keys
    # - Initialize vector store
    # - Load any cached data
    
    return config, logger


if __name__ == "__main__":
    # Initialize application
    config, logger = initialize_app()
    
    # Run Streamlit app
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        sys.exit(1)
