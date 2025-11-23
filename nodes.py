import random
import numpy as np
import math
import comfy
from nodes import common_ksampler, VAEDecodeTiled, ImageScaleBy, VAEEncodeTiled
from comfy.samplers import KSampler


class TagRandomizer:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags": ("STRING", {
                    "multiline": True,
                }),
                "delimiter": ("STRING", {"default": ","}),
                "min_power": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1}),
                "max_power": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 2.0, "step": 0.1}),
                "num_tags": ("INT", {"default": 5, "min": 1, "max": 200, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "mode": (["weighted", "normal"], {"default": "weighted"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "randomize_tags"
    CATEGORY = "text processing"

    def randomize_tags(self, tags, delimiter, min_power, max_power, num_tags, seed, mode):
        # Parse tags and clean up
        tag_list = [tag.strip() for tag in tags.split(delimiter) if tag.strip()]

        if not tag_list:
            return (",")

        # Set seed for reproducibility
        random.seed(seed)

        # Random selection without replacement
        selected_tags = random.sample(tag_list, min(num_tags, len(tag_list)))

        if mode == "weighted":
            # Assign random weights to each tag
            result_tags = []
            for tag in selected_tags:
                weight = round(random.uniform(min_power, max_power), 2)
                result_tags.append(f"({tag}:{weight})")
            result = ", ".join(result_tags)
        else:
            # Normal mode - just tags
            result = ", ".join(selected_tags)

        return (result,)
    

class StagedKsampler:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    upscale_shape = ["linear", "exponential", "inverse_exponential"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "vae": ("VAE",),
                "cfg_init": ("FLOAT", {"default": 4, "min": 0.0, "max": 100.0}),
                "cfg_step": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 100.0}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "upscale_method": (cls.upscale_methods,),
                "initial_step": ("INT", {"default": 15, "min": 1, "max": 10000, "step": 1}),
                "each_step_after": ("INT", {"default": 5, "min": 1, "max": 10000, "step": 1}),
                "iteration_count": ("INT", {"default": 3, "min": 1, "max": 10000, "step": 1}),
                "upscale_percentage_init": ("FLOAT", {"default": 1.6, "min": 0.0, "max": 100.0}),
                "upscale_percentage_end": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 100.0}),
                "upscale_percentage_shape": (cls.upscale_shape,),
                "initial_sampler_name": (KSampler.SAMPLERS,), 
                "sampler_name": (KSampler.SAMPLERS,), 
                "scheduler": (KSampler.SCHEDULERS,),
            }
        }

    RETURN_TYPES = ("LATENT",)
    
    FUNCTION     = "main"
    CATEGORY     = "Yearning-Nodes/samplers"
    EXPERIMENTAL = True

    def create_exponential_sequence(self, start, end, num_points, base=10):
        log_start = np.log(start) / np.log(base)
        log_end = np.log(end) / np.log(base)
        log_seq = np.linspace(log_start, log_end, num_points)
        return np.round(base ** log_seq, 2)
    
    def create_inverse_exponential(self, start, end, num_points, base=10):
        t = np.linspace(0, 1, num_points)
        normalized_seq = 1 - np.exp(-t * math.log(base))
        normalized_seq = normalized_seq / normalized_seq[-1]  
        
        return [round(start + (end - start) * val, 2) for val in normalized_seq]

    def main(self,
            model = None,
            positive = None,
            negative = None,
            latent_image = None,
            noise_seed = 0,
            vae = None,
            cfg_init = 4.0,
            cfg_step = 0.2,
            upscale_method: str = "lanczos",
            initial_step: int = 15,
            each_step_after: int = 5,
            iteration_count: int = 3,
            upscale_percentage_init: float = 1.6,
            upscale_percentage_end: float = 1.2,
            upscale_percentage_shape: str = "exponential",
            initial_sampler_name = None,
            sampler_name = None,
            scheduler = None,
            ):

        total_steps = initial_step + iteration_count * each_step_after
        vaedecoder = VAEDecodeTiled()
        vaeencoder = VAEEncodeTiled()
        scaler = ImageScaleBy()
        cfgs = [cfg_init + i * cfg_step for i in range(iteration_count)]
        np.random.seed(noise_seed)

        if upscale_percentage_init == upscale_percentage_end:
            upscales = [upscale_percentage_init for i in range(iteration_count)]

        else:
            if upscale_percentage_shape == "linear":
                self.upscales = np.linspace(
                    upscale_percentage_init, 
                    upscale_percentage_end, 
                    iteration_count
                )
            elif upscale_percentage_shape == "exponential":
                print(1)
                self.upscales = self.create_exponential_sequence(
                    upscale_percentage_init,
                    upscale_percentage_end,
                    iteration_count
                )
            elif upscale_percentage_shape == "inverse_exponential":
                self.upscales = self.create_inverse_exponential(
                    upscale_percentage_init,
                    upscale_percentage_end,
                    iteration_count
                )

        if iteration_count == 1:
            latent_image = common_ksampler(cfg=cfg_init,
                                            denoise=1,
                                            latent=latent_image,
                                            positive=positive,
                                            negative=negative,
                                            sampler_name=initial_sampler_name,
                                            scheduler=scheduler,
                                            model=model,
                                            seed=noise_seed,
                                            steps=total_steps,
                                            start_step=0,
                                            last_step=initial_step,
                                            force_full_denoise=True,
                                            disable_noise=False)[0]
            return (latent_image,)
            
        else:
            for i in range(iteration_count):
                if i == 0:
                    latent_image = common_ksampler(cfg=cfg_init,
                                                denoise=1,
                                                latent=latent_image,
                                                positive=positive,
                                                negative=negative,
                                                sampler_name=initial_sampler_name,
                                                scheduler=scheduler,
                                                model=model,
                                                seed=noise_seed,
                                                steps=total_steps,
                                                start_step=0,
                                                last_step=initial_step,
                                                force_full_denoise=True,
                                                disable_noise=False)[0]
                    
                    image = vaedecoder.decode(vae=vae, samples=latent_image, tile_size=512)[0]
                    resized = scaler.upscale(image=image, upscale_method=upscale_method, scale_by=self.upscales[i])[0]
                    latent_image = vaeencoder.encode(vae=vae, pixels=resized, tile_size=512, overlap=64)[0]
                    continue

                if i == iteration_count - 1:
                    latent_image = common_ksampler(cfg=cfg_init + cfg_step * i,
                                                denoise=1,
                                                latent=latent_image,
                                                positive=positive,
                                                negative=negative,
                                                sampler_name=sampler_name,
                                                scheduler=scheduler,
                                                model=model,
                                                seed=noise_seed,
                                                steps=total_steps,
                                                start_step=initial_step + each_step_after * (i-1),
                                                last_step=10000,
                                                force_full_denoise=True,
                                                disable_noise=False)[0]
                    
                    image = vaedecoder.decode(vae=vae, samples=latent_image, tile_size=512)[0]
                    resized = scaler.upscale(image=image, upscale_method=upscale_method, scale_by=self.upscales[i])[0]
                    latent_image = vaeencoder.encode(vae=vae, pixels=resized, tile_size=512, overlap=64)[0]
                    return (latent_image,)

                latent_image = common_ksampler(cfg=cfg_init + cfg_step * i,
                                            denoise=1,
                                            latent=latent_image,
                                            positive=positive,
                                            negative=negative,
                                            sampler_name=sampler_name,
                                            scheduler=scheduler,
                                            model=model,
                                            seed=noise_seed,
                                            steps=total_steps,
                                            start_step=initial_step + each_step_after * (i-1),
                                            last_step=initial_step + each_step_after * i,
                                            force_full_denoise=True,
                                            disable_noise=False)[0]
                        
                image = vaedecoder.decode(vae=vae, samples=latent_image, tile_size=512)[0]
                resized = scaler.upscale(image=image, upscale_method=upscale_method, scale_by=self.upscales[i])[0]
                latent_image = vaeencoder.encode(vae=vae, pixels=resized, tile_size=512, overlap=64)[0]


# Register nodes
NODE_CLASS_MAPPINGS = {
    "TagRandomizer": TagRandomizer,
    "StagedKSampler": StagedKsampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TagRandomizer": "Tag Randomizer",
    "StagedKSampler": "Staged KSampler"
}