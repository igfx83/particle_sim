import json
import logging
from pathlib import Path


def load_elements(elements_dir="data/elements"):
    elements = {}
    elements_path = Path(elements_dir)
    if not elements_path.exists():
        logging.error(f"Elements data directory not found: {elements_dir}")
        return elements

    for file_path in elements_path.glob("*.json"):
        try:
            with file_path.open("r") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError(
                        f"File {file_path.name} does not contain a dictionary"
                    )
                for key in data.keys():
                    if not isinstance(key, str):
                        raise ValueError(
                            f"Non-string key found in {file_path.name}: {key}"
                        )
                logging.info(f"Loaded from {file_path.name}: {list(data.keys())}")
                elements.update(data)
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Error in {file_path.name}: {e}")
    logging.info(f"ELEMENTS keys: {list(elements.keys())}")
    return elements
