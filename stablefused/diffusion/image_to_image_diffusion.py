import numpy as np
import torch

from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import List, Optional, Union

from .base_diffusion import BaseDiffusion


class ImageToImageDiffusion(BaseDiffusion):
    def __init__(
        self,
        model_id: str = None,
        tokenizer: CLIPTokenizer = None,
        text_encoder: CLIPTextModel = None,
        vae: AutoencoderKL = None,
        unet: UNet2DConditionModel = None,
        scheduler: KarrasDiffusionSchedulers = None,
        device="cuda",
    ) -> None:
        super().__init__(
            model_id, tokenizer, text_encoder, vae, unet, scheduler, device
        )

    def prompt_to_embedding(
        self,
        prompt: Union[str, List[str]],
        guidance_scale: float,
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> torch.FloatTensor:
        """Convert prompt(s) to a CLIP embedding(s)."""

        use_classifier_free_guidance = guidance_scale > 1.0

        if negative_prompt is not None:
            assert type(prompt) is type(negative_prompt)

        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
            if negative_prompt is not None:
                negative_prompt = [negative_prompt]
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise TypeError("`prompt` must be a string or a list of strings")

        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_input.attention_mask.to(self.device)
        else:
            attention_mask = None

        text_embedding = self.text_encoder(
            text_input.input_ids.to(self.device), attention_mask=attention_mask
        )[0]

        if use_classifier_free_guidance:
            if negative_prompt is None:
                unconditioning_input = [""] * batch_size
            else:
                unconditioning_input = negative_prompt

            unconditioning_input = self.tokenizer(
                unconditioning_input,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            unconditional_embedding = self.text_encoder(
                unconditioning_input.input_ids.to(self.device),
                attention_mask=attention_mask,
            )[0]

        embedding = torch.cat([unconditional_embedding, text_embedding])
        return embedding

    def embedding_to_latent(
        self,
        embedding: torch.FloatTensor,
        image: Image.Image,
        num_inference_steps: int,
        start_step: int,
        guidance_scale: float,
        latent: Optional[torch.FloatTensor] = None,
        return_latent_history: bool = False,
    ) -> Union[torch.FloatTensor, List[torch.FloatTensor]]:
        use_classifier_free_guidance = guidance_scale > 1.0

        if latent is None:
            latent = self.image_to_latent(image)
        latent = latent.to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        if start_step > 0:
            start_timestep = (
                self.scheduler.timesteps[start_step].repeat(latent.shape[0]).long()
            )
            noise = torch.randn(latent.shape).to(self.device)
            latent = self.scheduler.add_noise(latent, noise, start_timestep)

        timesteps = self.scheduler.timesteps[start_step:]
        latent_history = [latent]

        for i, timestep in tqdm(enumerate(timesteps)):
            latent_model_input = (
                torch.cat([latent] * 2) if use_classifier_free_guidance else latent
            )
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep
            )

            noise_prediction = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=embedding,
                return_dict=False,
            )[0]

            if use_classifier_free_guidance:
                noise_unconditional, noise_prompt = noise_prediction.chunk(2)
                noise_prediction = noise_unconditional + guidance_scale * (
                    noise_prompt - noise_unconditional
                )

            latent = self.scheduler.step(
                noise_prediction, timestep, latent, return_dict=False
            )[0]

            if return_latent_history:
                latent_history.append(latent)

        return latent_history if return_latent_history else latent

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        prompt: Optional[Union[str, List[str]]],
        num_inference_steps: int = 50,
        start_step: int = 0,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        output_type: str = "pil",
        return_latent_history: bool = False,
    ) -> Union[torch.Tensor, np.ndarray, List[Image.Image]]:
        """Generate image(s) from image(s)."""

        self.validate_input(
            prompt=prompt,
            negative_prompt=negative_prompt,
            start_step=start_step,
            num_inference_steps=num_inference_steps,
        )
        embedding = self.prompt_to_embedding(prompt, guidance_scale, negative_prompt)
        latent = self.embedding_to_latent(
            embedding=embedding,
            image=image,
            num_inference_steps=num_inference_steps,
            start_step=start_step,
            guidance_scale=guidance_scale,
            return_latent_history=return_latent_history,
        )

        if output_type == "latent":
            return latent

        if return_latent_history:
            image: np.ndarray = np.array(
                [self.latent_to_image(l, output_type) for l in tqdm(latent)]
            )
            dims = len(image.shape)
            image = np.transpose(image, (1, 0, *range(2, dims)))
        else:
            image = self.latent_to_image(latent, output_type)
        return image
