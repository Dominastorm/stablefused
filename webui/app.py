import numpy as np
import streamlit as st
import torch

from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_option_menu import option_menu
from streamlit_tags import st_tags
from diffusers import DPMSolverMultistepScheduler

from stablefused.utils import image_grid, pil_to_video
from stablefused import (
    ImageToImageDiffusion,
    LatentWalkDiffusion,
    TextToImageDiffusion,
    InpaintDiffusion,
)

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
        with st.spinner("Generating..."):
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

    seed = st.number_input(
        label="Seed",
        min_value=0,
        max_value=2**32,
        value=1337,
        step=1,
        key="seed",
        format="%d",
    )

    interpolation_steps = st.number_input(
        label="Interpolation Steps",
        min_value=1,
        max_value=100,
        value=24,
        step=1,
        key="interpolation_steps",
    )

    fps = st.number_input(
        label="FPS",
        min_value=1,
        max_value=60,
        value=5,
        step=1,
        key="fps",
        format="%d",
    )

    if "story" not in st.session_state:
        st.session_state.story = []

    story_prompt = st.text_input(
        label="Story Prompt",
        value="",
    )

    col1, col2, col3 = st.columns(3)
    if col1.button("Add Prompt"):
        st.session_state.story.append(story_prompt)

    if col2.button("Clear Story"):
        st.session_state.story = []

    if col3.button("Use Sample Story"):
        st.session_state.story = [
            "A dog chasing a cat in a thrilling backyard scene, high quality and photorealistic",
            "A determined dog in hot pursuit, with stunning realism, octane render",
            "A thrilling chase, dog behind the cat, octane render, exceptional realism and quality",
            "The exciting moment of a cat outmaneuvering a chasing dog, high-quality and photorealistic detail",
            "A clever cat escaping a determined dog and soaring into space, rendered with octane render for stunning realism",
            "The cat's escape into the cosmos, leaving the dog behind in a scene,high quality and photorealistic style",
        ]
    st.write(st.session_state.story)

    if st.button("Generate"):
        with st.spinner("Generating..."):
            lw_model = LatentWalkDiffusion(model_id=model_id)

            lw_model.enable_attention_slicing()
            lw_model.enable_slicing()
            lw_model.enable_tiling()

            torch.manual_seed(seed)
            np.random.seed(seed)

            # t2i_model = TextToImageDiffusion(
            #     model_id=model_id, torch_dtype=torch.float16
            # )

            shape = (
                1,
                lw_model.unet.config.in_channels,
                image_height // lw_model.vae_scale_factor,
                image_width // lw_model.vae_scale_factor,
            )
            single_latent = lw_model.random_tensor(shape)
            latent = single_latent.repeat(num_images, 1, 1, 1)

            # t2i_images = t2i_model(
            #     prompt=story_prompt,
            #     num_inference_steps=20,
            #     guidance_scale=15.0,
            #     latent=latent,
            # )

            story = st.session_state.story
            story_images = []
            for i in range(len(story) - 1):
                current_prompt = story[i : i + 2]
                current_latent = latent[i : i + 2]
                imgs = lw_model.interpolate(
                    prompt=current_prompt,
                    negative_prompt=[negative_prompt] * len(current_prompt),
                    latent=current_latent,
                    num_inference_steps=num_inference_steps,
                    interpolation_steps=interpolation_steps,
                )
                story_images.extend(imgs)

            path = "diffusion_story.mp4"
            pil_to_video(story_images, path, fps=fps)

            st.video(open(path, "rb").read())


def display_inpaint():
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
    )

    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    if drawing_mode == "point":
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    start_image = st.sidebar.file_uploader("Start image:", type=["png", "jpg"])

    prompt = st.text_input(
        label="Prompt",
        value="A painting of a cat",
    )

    negative_prompt = st.text_input(
        label="Negative Prompt",
        value="cartoon, unrealistic, blur, boring background, deformed, disfigured, low resolution, unattractive",
    )

    num_images = st.number_input(
        label="Number of Output Images",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        key="num_images",
    )

    canvas_result = st_canvas(
        # Fixed fill color with some opacity
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color="white",
        background_image=Image.open(start_image) if start_image else None,
        update_streamlit=True,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == "point" else 0,
        key="canvas",
    )

    if st.button("Generate"):
        if canvas_result.image_data is None:
            mask = np.zeros((512, 512))
            mask = Image.fromarray(mask).convert("L")
        else:
            mask = Image.fromarray(canvas_result.image_data).convert("L")
            mask = np.array(mask)
            mask[mask > 0] = 255
            mask = Image.fromarray(mask)

        st.write(np.array(mask))
        st.image(mask, clamp=True)

        start_image = Image.open(start_image).convert("RGB")
        start_image = np.array(start_image)
        start_image = Image.fromarray(start_image)
        st.image(start_image)

        model = InpaintDiffusion(
            model_id="runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
        )
        model.enable_attention_slicing()
        model.enable_slicing()
        model.enable_tiling()

        images = model(
            prompt=[prompt] * num_images,
            negative_prompt=[negative_prompt] * num_images,
            image=start_image,
            mask=mask,
            num_inference_steps=2,
            start_step=0,
            image_height=512,
            image_width=512,
            guidance_scale=10.0,
        )

        images = image_grid(images, rows=1, cols=num_images)
        st.image(images, clamp=True)


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
