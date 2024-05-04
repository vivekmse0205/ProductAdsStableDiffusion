import os
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


class ProductGenerator:
    def __init__(self, base_model, repo, checkpoint, output_path, prompt= {}, num_inference_steps=8,num_images_per_prompt=5):
        self.base_model = base_model
        self.repo = repo
        self.checkpoint = checkpoint
        self.output_path = output_path
        self.num_inference_steps = num_inference_steps
        self.product_prompt = prompt
        self.num_images_per_prompt = num_images_per_prompt

    def load_model(self):
        print('Loading model',self.base_model,self.repo,self.checkpoint)
        unet = UNet2DConditionModel.from_config(self.base_model, subfolder="unet").to("cuda", torch.float16)
        unet.load_state_dict(load_file(hf_hub_download(self.repo, self.checkpoint), device="cuda"))
        print('Setting pipeline')
        self.pipe = StableDiffusionXLPipeline.from_pretrained(self.base_model, unet=unet, torch_dtype=torch.float16,
                                                              variant="fp16").to("cuda")
        # Ensure sampler uses "trailing" timesteps.
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config,
                                                                 timestep_spacing="trailing")

    def generate_images(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        for product, prompts in self.product_prompt.items():
            print(f"Processing product {product}")
            for prompt_id, prompt in enumerate(prompts):
                images = self.pipe(prompt, num_inference_steps=self.num_inference_steps, guidance_scale=0, num_images_per_prompt=self.num_images_per_prompt).images
                for img_id, img in enumerate(images):
                    img_name = f"{product}_{prompt_id}_{img_id}.png"
                    save_path = os.path.join(self.output_path, img_name)
                    img.save(save_path)


def main():
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    checkpoint = "sdxl_lightning_4step_unet.safetensors"
    output_path = "output"

    processor = ProductGenerator(base, repo, checkpoint, output_path)
    processor.load_model()
    processor.process_images()


if __name__ == "__main__":
    main()
