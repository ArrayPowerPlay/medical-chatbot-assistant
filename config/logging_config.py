import logging
import sys

def setup_logging():
    logging.basicConfig(      # Global config
        level=logging.INFO,   # Log from INFO level or above (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),     # Log to terminal
            logging.FileHandler("project.log")     # Log to file, support for debugging
        ]
    )
    # Turn off unnecessary logs of external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)   # Log from WARNING level or above
    logging.getLogger("neo4j").setLevel(logging.WARNING)


logger = logging.getLogger("MedKG-RAG")