import logging
import os
from logging.config import fileConfig

import yaml

from configs.product_gen_prompt_config import product_prompt_v1
from src.product_enhancer.enhance_product import ProductEnhancer
from src.product_visuals.generate_products import ProductGenerator

# Load logging configuration from file
# fileConfig('configs/logging_config.ini')
# logger = logging.getLogger(__name__)


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Get the path to the configs file relative to the location of main.py
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'config.yml')
    config = load_config(config_path)
    # Load visual generator model
    vg_base = config["visual_generator_model"]["base"]
    vg_repo = config["visual_generator_model"]["repo"]
    vg_ckpt = config["visual_generator_model"]["checkpoint"]
    output_path = config["product_output_path"]
    det_save_path = config["detection_save_path"]
    seg_save_path = config["segmentation_save_path"]
    impaint_save_path = config["impaint_save_path"]
    detector_repo = config["detector_model_repo"]
    sam_repo = config["sam_model_repo"]
    impaint_repo = config["impaint_model_repo"]
    processor = ProductGenerator(vg_base, vg_repo, vg_ckpt, output_path, prompt=product_prompt_v1)
    processor.load_model()
    processor.generate_images()
    prompt = ". ".join(product_prompt_v1.keys()) + "."
    product_enhancer = ProductEnhancer(detector_repo, sam_repo, impaint_repo, output_path, det_save_path, seg_save_path,
                                       impaint_save_path)
    product_enhancer.enhance_images(prompt)


if __name__ == "__main__":
    main()
