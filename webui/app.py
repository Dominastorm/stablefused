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
        "Text to Image",
        "Image to Image",
        "Latent Walk",
        "Inpaint",
        "Inpaint Walk",
    ],
    # icons from https://icons.getbootstrap.com/
    icons=["fonts", "card-image", "arrows-move", "brush", "palette"],
    default_index=0,
    orientation=orientation,
)


def display_text_to_image():
    import numpy as np
    import torch

    from IPython.display import Video, display
    from stablefused import TextToImageDiffusion
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

    num_outputs = st.number_input(
        label="Number of Output Images",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key="num_outputs",
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
            format="%d"
        )

    
    if st.button("Generate"):
        with st.spinner("Generating image..."):
            model = TextToImageDiffusion(model_id=model_id)
            model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)
            model.enable_attention_slicing()
            model.enable_slicing()
            model.enable_tiling()

            torch.manual_seed(seed)
            np.random.seed(seed)

            images = model(
                prompt=[prompt] * num_outputs,
                negative_prompt=[negative_prompt] * num_outputs,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                return_latent_history=generate_video,
            )

            if generate_video:
                timestep_images = []
                for imgs in zip(*images):
                    img = image_grid(imgs, rows=1, cols=num_outputs)
                    timestep_images.append(img)

                path = "text_to_image_diffusion.mp4"
                pil_to_video(timestep_images, path, fps=5)
                st.video(open(path, "rb").read())
            else:
                images = image_grid(images, rows=1, cols=num_outputs)
                st.image(images, clamp=True)

def display_image_to_image():
    ...


def display_latent_walk():
    ...


def display_inpaint():
    ...


def display_inpaint_walk():
    ...


def display_page():
    match current_tab:
        case "Text to Image":
            display_text_to_image()
        case "Image to Image":
            display_image_to_image()
        case "Latent Walk":
            display_latent_walk()
        case "Inpaint":
            display_inpaint()
        case "Inpaint Walk":
            display_inpaint_walk()


display_page()
