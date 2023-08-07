import numpy as np
import torch
import pytest

from stablefused import BaseDiffusion, ImageToImageDiffusion


@pytest.fixture
def model():
    """
    Fixture to initialize the ImageToImageDiffusion model and set random seeds for reproducibility.

    Returns
    -------
    ImageToImageDiffusion
        The initialized ImageToImageDiffusion model.
    """
    seed = 1337
    model_id = "hf-internal-testing/tiny-stable-diffusion-pipe"
    device = "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ImageToImageDiffusion(model_id=model_id, device=device)
    return model


@pytest.fixture
def config():
    return {
        "prompt": "a photo of a cat",
        "num_inference_steps": 2,
        "start_step": 1,
        "image_dim": 64,
    }


def test_image_to_image_diffusion_1(model: ImageToImageDiffusion, config: dict):
    """
    Test case for the ImageToImageDiffusion model.

    Parameters
    ----------
    model
        The initialized ImageToImageDiffusion model fixture.
    config
        The configuration dictionary for the test case.

    Raises
    ------
    AssertionError
        If the generated image is not of type np.ndarray.
        If the generated image does not have the expected shape.
    """
    dim = config.get("image_dim")
    image = torch.randn(1, 3, dim, dim)

    images = model(
        image=image,
        prompt=config.get("prompt"),
        num_inference_steps=config.get("num_inference_steps"),
        output_type="np",
    )

    assert type(images) is np.ndarray
    assert images.shape == (1, dim, dim, 3)


def test_image_to_image_diffusion_2(model: ImageToImageDiffusion, config: dict):
    """
    Test case for the ImageToImageDiffusion model.

    Parameters
    ----------
    model
        The initialized ImageToImageDiffusion model fixture.
    config
        The configuration dictionary for the test case.

    Raises
    ------
    AssertionError
        If the generated image is not of type np.ndarray.
        If the generated image does not have the expected shape.
    """
    dim = config.get("image_dim")
    image = torch.randn(1, 3, dim, dim)

    images = model(
        image=image,
        prompt=config.get("prompt"),
        num_inference_steps=config.get("num_inference_steps"),
        start_step=config.get("start_step"),
        output_type="pt",
        return_latent_history=True,
    )

    assert type(images) is np.ndarray
    assert images.shape == (
        1,
        config.get("num_inference_steps") + 1 - config.get("start_step"),
        3,
        dim,
        dim,
    )


if __name__ == "__main__":
    pytest.main([__file__])