import os
import pathlib

import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import SamModel, SamProcessor

from . import utils


class ProductEnhancer:
    def __init__(self, detector_model, segmentor_model, impaint_model, images_path,
                 detection_save_path="detection_results", segmentation_save_path="seg_results",
                 impaint_save_path="impaint_results"):
        self.detector_model_id = detector_model
        self.segmentor_model_id = segmentor_model
        self.impaint_model_id = impaint_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # detector initialization
        self.detector_processor = AutoProcessor.from_pretrained(self.detector_model_id)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(self.detector_model_id).to(self.device)
        self.detection_save_path = detection_save_path
        os.makedirs(self.detection_save_path,exist_ok=True)
        # Segmentation initialization
        self.segmentation_processor = SamProcessor.from_pretrained(self.segmentor_model_id)
        self.segmention_model = SamModel.from_pretrained(self.segmentor_model_id).to(self.device)
        self.segmentation_save_path = segmentation_save_path
        os.makedirs(self.segmentation_save_path,exist_ok=True)
        # Impaint initialization
        self.impaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.impaint_model_id, torch_dtype=torch.float16
        ).to(self.device)
        self.impaint_save_path = impaint_save_path
        os.makedirs(self.impaint_save_path,exist_ok=True)
        self.images_base_path = images_path
        self.images_list = sorted(list(os.listdir(self.images_base_path)))
        self.current_image_name = None
        self.current_image = None
        self.boxes = []
        self.labels = None
        self.masks = None
        self.background_enhancer = False

    def impaint_image(self, image, masks, idx2label, prompt=""):
        img = image.resize((512, 512))
        for idx, mask in enumerate(masks):
            mask_image = Image.fromarray(mask, mode="L")
            mask_image = mask_image.resize((512, 512))
            if not prompt:
                prompt = f"Correct any uneven lighting on the {idx2label[idx]} to uniformly highlight its features. Make {idx2label[idx]} more realistic and remove artifacts if any "
            impaint_image = self.impaint_pipe(prompt=prompt, image=img, mask_image=mask_image).images[0]
            impaint_image.save(f"{self.impaint_save_path}/{self.current_image_name}_{idx}_{idx2label[idx]}.png")
        return True

    def segment_objects(self, image, boxes):
        sam_inputs = self.segmentation_processor(image, input_boxes=boxes, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.segmention_model(**sam_inputs)
        masks = self.segmentation_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(), sam_inputs["reshaped_input_sizes"].cpu()
        )
        return masks

    def detect_objects(self, image, prompt):
        inputs = self.detector_processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.detector(**inputs)
        results = self.detector_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        # Hardcoding [0] as the assumption is we process single image at a given time
        result = results[0]
        boxes = result["boxes"].cpu().tolist()
        labels = result["labels"]
        return boxes, labels

    def enhance_image(self, image_path, prompt="", save_visuals=True):
        self.current_image = Image.open(image_path)
        self.current_image_name = pathlib.Path(image_path).stem
        self.boxes, self.labels = self.detect_objects(self.current_image, prompt)
        det_image = self.current_image.copy()
        if not self.boxes:
            return False
        for box in self.boxes:
            det_image = utils.draw_box_on_image(det_image, *box)
        if save_visuals:
            bbox_save_path = os.path.join(self.detection_save_path, f"{self.current_image_name}.png")
            det_image.save(bbox_save_path)
        self.masks = self.segment_objects(self.current_image, [self.boxes])
        self.masks, label2idx, idx2label = utils.combine_masks(self.labels, self.masks[0])
        if save_visuals:
            for idx, mask in enumerate(self.masks):
                mask_image = Image.fromarray(mask, mode="L")
                mask_save_path = os.path.join(self.segmentation_save_path,
                                              f"{self.current_image_name}_{idx2label[idx]}.png")
                # Save the mask as a PNG file
                mask_image.save(mask_save_path)
        self.impaint_image(self.current_image, self.masks, idx2label)

    def enhance_images(self, text_prompt):
        for img_idx, image_name in enumerate(self.images_list):
            image_path = os.path.join(self.images_base_path, image_name)
            self.enhance_image(image_path, text_prompt)

