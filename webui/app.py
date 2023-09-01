import random
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_tags import st_tags

from diffusers import DPMSolverMultistepScheduler


page_config = st.set_page_config(
    page_title="Stablefused",
    page_icon="🧊",
    layout="wide",
)

st_style = """<style> footer {visibility: hidden;} </style>"""
st.markdown(st_style, unsafe_allow_html=True)

orientation = "horizontal"

current_tab = option_menu(
    menu_title="Stablefused",
    menu_icon="box",
    options=[
        "Image Gen",
        "Video Gen",
        "Latent Walk",
        "Inpaint",
        "Inpaint Walk",
    ],
    # icons from https://icons.getbootstrap.com/
    icons=["card-image", "film", "arrows-move", "brush", "palette"],
    default_index=0,
    orientation=orientation,
)


def display_image_gen():
    import numpy as np
    import torch

    from PIL import Image
    from stablefused import TextToImageDiffusion, ImageToImageDiffusion
    from stablefused.utils import image_grid, pil_to_video

    model_id = st.selectbox(
        label="Select model",
        options=[
            "runwayml/stable-diffusion-v1-5",
            "CompVis/stable-diffusion-v1-4",
            "stabilityai/stable-diffusion-2-1-base",
        ],
    )

    prompt = st.text_input(
        label="Prompt",
        value="A painting of a cat",
    )

    negative_prompt = st.text_input(
        label="Negative Prompt",
        value="cartoon, unrealistic, blur, boring background, deformed, disfigured, low resolution, unattractive",
    )

    start_image = st.file_uploader(
        label="Start Image",
        type=["png", "jpg", "jpeg"],
    )

    image_height = st.number_input(
        label="Image Height",
        min_value=16,
        max_value=4096,
        value=512,
        step=8,
        key="image_height",
        format="%d",
    )

    image_width = st.number_input(
        label="Image Width",
        min_value=16,
        max_value=4096,
        value=512,
        step=8,
        key="image_width",
    )

    num_images = st.number_input(
        label="Number of Output Images",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key="num_images",
    )

    num_inference_steps = st.number_input(
        label="Number of Inference Steps",
        min_value=1,
        max_value=100,
        value=20,
        step=1,
        key="num_inference_steps",
    )

    guidance_scale = st.number_input(
        label="Guidance Scale",
        min_value=0.0,
        max_value=50.0,
        value=7.5,
        step=0.1,
        key="guidance_scale",
    )

    guidance_rescale = st.number_input(
        label="Guidance Rescale",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.01,
        key="guidance_rescale",
    )

    seed = st.number_input(
        label="Seed",
        min_value=0,
        max_value=2**32,
        value=1337,
        step=1,
        key="seed",
        format="%d",
    )

    generate_video = st.checkbox(
        label="Generate Video",
        value=False,
        key="generate_video",
    )

    if generate_video:
        fps = st.number_input(
            label="FPS",
            min_value=1,
            max_value=60,
            value=5,
            step=1,
            key="fps",
            format="%d",
        )

    if st.button("Generate"):
        with st.spinner("Generating image..."):
            if start_image:
                model = ImageToImageDiffusion(model_id=model_id)
                start_image = [Image.open(start_image)] * num_images
            else:
                model = TextToImageDiffusion(model_id=model_id)

            model.scheduler = DPMSolverMultistepScheduler.from_config(
                model.scheduler.config
            )
            model.enable_attention_slicing()
            model.enable_slicing()
            model.enable_tiling()

            torch.manual_seed(seed)
            np.random.seed(seed)

            if start_image:
                images = model(
                    image=start_image,
                    prompt=[prompt] * num_images,
                    negative_prompt=[negative_prompt] * num_images,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    guidance_rescale=guidance_rescale,
                    return_latent_history=generate_video,
                )
            else:
                images = model(
                    prompt=[prompt] * num_images,
                    negative_prompt=[negative_prompt] * num_images,
                    num_inference_steps=num_inference_steps,
                    image_height=image_height,
                    image_width=image_width,
                    guidance_scale=guidance_scale,
                    guidance_rescale=guidance_rescale,
                    return_latent_history=generate_video,
                )

            if generate_video:
                timestep_images = []
                for imgs in zip(*images):
                    img = image_grid(imgs, rows=1, cols=num_images)
                    timestep_images.append(img)

                path = "diffusion_latent_history.mp4"
                pil_to_video(timestep_images, path, fps=fps)
                st.video(open(path, "rb").read())
            else:
                images = image_grid(images, rows=1, cols=num_images)
                st.image(images, clamp=True)


def display_video_gen():
    ...


def display_latent_walk():
    ...


def display_inpaint():
    ...


def display_inpaint_walk():
    ...


def display_page():
    match current_tab:
        case "Image Gen":
            display_image_gen()
        case "Video Gen":
            display_video_gen()
        case "Latent Walk":
            display_latent_walk()
        case "Inpaint":
            display_inpaint()
        case "Inpaint Walk":
            display_inpaint_walk()


display_page()
