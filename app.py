import gradio as gr
import torch
import requests
from io import BytesIO
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from utils import *
from inversion_utils import *
from torch import autocast, inference_mode
import re


def invert(x0, prompt_src="", num_diffusion_steps=100, cfg_scale_src = 3.5, eta = 1):

  #  inverts a real image according to Algorihm 1 in https://arxiv.org/pdf/2304.06140.pdf, 
  #  based on the code in https://github.com/inbarhub/DDPM_inversion
   
  #  returns wt, zs, wts:
  #  wt - inverted latent
  #  wts - intermediate inverted latents
  #  zs - noise maps

  sd_pipe.scheduler.set_timesteps(num_diffusion_steps)

  # vae encode image
  with autocast("cuda"), inference_mode():
      w0 = (sd_pipe.vae.encode(x0).latent_dist.mode() * 0.18215).float()

  # find Zs and wts - forward process
  wt, zs, wts = inversion_forward_process(sd_pipe, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src, prog_bar=False, num_inference_steps=num_diffusion_steps)
  return wt, zs, wts



def sample(wt, zs, wts, prompt_tar="", cfg_scale_tar=15, skip=36, eta = 1):

    # reverse process (via Zs and wT)
    w0, _ = inversion_reverse_process(sd_pipe, xT=wts[skip], etas=eta, prompts=[prompt_tar], cfg_scales=[cfg_scale_tar], prog_bar=False, zs=zs[skip:])
    
    # vae decode image
    with autocast("cuda"), inference_mode():
        x0_dec = sd_pipe.vae.decode(1 / 0.18215 * w0).sample
    if x0_dec.dim()<4:
        x0_dec = x0_dec[None,:,:,:]
    img = image_grid(x0_dec)
    return img

# load pipelines
# sd_model_id = "runwayml/stable-diffusion-v1-5"
# sd_model_id = "CompVis/stable-diffusion-v1-4"
sd_model_id = "stabilityai/stable-diffusion-2-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_id).to(device)
sd_pipe.scheduler = DDIMScheduler.from_config(sd_model_id, subfolder = "scheduler")



def get_example():
    case = [
        [
            'Examples/gnochi_mirror.jpeg', 
            '',
            'watercolor painting of a cat sitting next to a mirror',
            100,
            3.5,
            36,
            15,
            'Examples/gnochi_mirror_watercolor_painting.png', 
             ],]
    return case


def edit(input_image, 
            src_prompt ="", 
            tar_prompt="",
            steps=100,
            cfg_scale_src = 3.5,
            cfg_scale_tar = 15,
            skip=36,
            wt = None,
            zs = None,
            wts = None
             
):
    torch.manual_seed(seed)
     # offsets=(0,0,0,0)
    x0 = load_512(input_image, left,right, top, bottom, device)

    if not wt:
        # invert and retrieve noise maps and latent
        wt, zs, wts = invert(x0 =x0 , prompt_src=src_prompt, num_diffusion_steps=steps, cfg_scale_src=cfg_scale_src)
    
    output = sample(wt, zs, wts, prompt_tar=tar_prompt, cfg_scale_tar=cfg_scale_tar, skip=skip)

    return output


def reset_latents():
    wt = gr.State(value=None)
    zs = gr.State(value=None)
    wts = gr.State(value=None)



########
# demo #
########
                        
intro = """
<h1 style="font-weight: 1400; text-align: center; margin-bottom: 7px;">
   Edit Friendly DDPM Inversion
</h1>
<p style="font-size: 0.9rem; text-align: center; margin: 0rem; line-height: 1.2em; margin-top:1em">
<a href="https://arxiv.org/abs/2301.12247" style="text-decoration: underline;" target="_blank">An Edit Friendly DDPM Noise Space:
Inversion and Manipulations </a> 
<p/>
<p style="font-size: 0.9rem; margin: 0rem; line-height: 1.2em; margin-top:1em">
For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings.
<a href="https://huggingface.co/spaces/LinoyTsaban/ddpm_sega?duplicate=true">
<img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
<p/>"""
with gr.Blocks() as demo:
    gr.HTML(intro)
    wt = gr.State(value=None)
    zs = gr.State(value=None)
    wts = gr.State(value=None)
    with gr.Row():
        input_image = gr.Image(label="Input Image", interactive=True)
        input_image.style(height=512, width=512)
        # inverted_image = gr.Image(label=f"Reconstructed Image", interactive=False)
        # inverted_image.style(height=512, width=512)
        output_image = gr.Image(label=f"Edited Image", interactive=False)
        output_image.style(height=512, width=512)
    
    with gr.Row():
        tar_prompt = gr.Textbox(lines=1, label="Describe the image yout want", interactive=True, placeholder="tip: use concepts from the original image for the description")

    with gr.Row():
        # with gr.Column(scale=1, min_width=100):
        #     invert_button = gr.Button("Invert")
        # with gr.Column(scale=1, min_width=100):
        #     edit_button = gr.Button("Edit")
        with gr.Column(scale=1, min_width=100):
            edit_button = gr.Button("Run")



    with gr.Accordion("Advanced Options", open=False):
        with gr.Row():
            with gr.Column():
                #inversion
                src_prompt = gr.Textbox(lines=1, label="Source Prompt", interactive=True, placeholder="optional: describe the original image")
                steps = gr.Number(value=100, precision=0, label="Num Diffusion Steps", interactive=True)
                cfg_scale_src = gr.Slider(minimum=1, maximum=15, value=3.5, label=f"Source Guidance Scale", interactive=True)
      
                # reconstruction
                skip = gr.Slider(minimum=0, maximum=40, value=36, precision=0, label="Skip Steps", interactive=True)
                cfg_scale_tar = gr.Slider(minimum=7, maximum=18,value=15, label=f"Target Guidance Scale", interactive=True)
                seed = gr.Number(value=0, precision=0, label="Seed", interactive=True)
            
          

    # gr.Markdown(help_text)

    # invert_button.click(
    #     fn=edit,
    #     inputs=[input_image, 
    #                 src_prompt, 
    #                 src_prompt,
    #                 steps,
    #                 cfg_scale_src,
    #                 cfg_scale_tar,
    #                 skip,
    #                 seed,
    #                 left,
    #                 right,
    #                 top,
    #                 bottom
    #     ],
    #     outputs = [inverted_image],
    # )

    edit_button.click(
        fn=edit,
        inputs=[input_image, 
            src_prompt, 
            tar_prompt,
            steps,
            cfg_scale_src,
            cfg_scale_tar,
            skip,
            seed,
            wt,
                zs,
                wts

        ],
        outputs=[output_image],
    )

    input_image.change(
        fn = reset_latents
    )


    gr.Examples(
        label='Examples', 
        examples=get_example(), 
        inputs=[input_image, src_prompt, tar_prompt, steps,
                    cfg_scale_tar,
                    skip,
                    cfg_scale_tar,
                    output_image
               ],
        outputs=[output_image ],
        # fn=edit,
        # cache_examples=True
    )



demo.queue()
demo.launch(share=False)