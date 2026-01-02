import json
import logging
from pathlib import Path

from engine.components.Constants import AIR_DENSITY


def load_elements(
    elements_dir="/data/data/com.termux/files/home/storage/shared/code/particle_sim/src/data/elements",
):
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
    elements = calculate_element_drag_coeff(elements)
    return elements


def calculate_element_drag_coeff(elements, scale_pixel_to_m=1e-3):
    """
    Compute drag coefficient for an element type.
    Call this once per element during element definition.
    """
    for name, element_props in elements.items():
        try:
            intrinsic_props = element_props.get("intrinsic_properties")

            mass = intrinsic_props["mass"]
            density = intrinsic_props["density"]
            radius = intrinsic_props["radius"]

            radius_m = radius * scale_pixel_to_m
            area = 3.14159 * radius_m * radius_m
            k_si = 0.5 * AIR_DENSITY * 0.47 * area
            k_pixel = k_si / (scale_pixel_to_m * scale_pixel_to_m * max(mass, 1e-6))

            elements[name]["intrinsic_properties"]["drag_coeff"] = k_pixel

        except KeyError:
            logging.error(
                f"An element is malformed or missing properties. Examine the definition for the element {name} and ensure is follows the correct interface."
            )

    return elements
