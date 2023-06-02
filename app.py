import gradio as gr
import torch
import random
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
  return zs, wts



def sample(zs, wts, prompt_tar="", skip=36, cfg_scale_tar=15, eta = 1):

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
with gr.Blocks(css='style.css') as demo:
    
    def reset_do_inversion():
        do_inversion = True
        return do_inversion


    def edit(input_image,
              do_inversion, 
            src_prompt ="", 
            tar_prompt="",
            steps=100,
            cfg_scale_src = 3.5,
            cfg_scale_tar = 15,
            skip=36,
            seed = 0,
            randomized_seed = True):

        if randomized_seed:
            seed = random.randint(0, np.iinfo(np.int32).max)
            
        torch.manual_seed(seed)
         # offsets=(0,0,0,0)
        x0 = load_512(input_image, device=device)
    
        if do_inversion:
            # invert and retrieve noise maps and latent
            zs_tensor, wts_tensor = invert(x0 =x0 , prompt_src=src_prompt, num_diffusion_steps=steps, cfg_scale_src=cfg_scale_src)
            # xt = gr.State(value=wts[skip])
            # zs = gr.State(value=zs[skip:])
            wts = gr.State(value=wts_tensor)
            zs = gr.State(value=zs_tensor)
            do_inversion = False
        
        # output = sample(zs.value, xt.value, prompt_tar=tar_prompt, cfg_scale_tar=cfg_scale_tar)
        output = sample(zs.value, wts.value, prompt_tar=tar_prompt, skip=skip, cfg_scale_tar=cfg_scale_tar)
    
        return output, wts, zs, do_inversion
    
    gr.HTML(intro)
    # xt = gr.State(value=False)
    wts = gr.State()
    zs = gr.State()
    do_inversion = gr.State(value=True)
    with gr.Row():
        input_image = gr.Image(label="Input Image", interactive=True)
        input_image.style(height=512, width=512)
        output_image = gr.Image(label=f"Edited Image", interactive=False)
        output_image.style(height=512, width=512)
    
    with gr.Row():
        tar_prompt = gr.Textbox(lines=1, label="Describe your desired edited output", interactive=True)

    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            edit_button = gr.Button("Run")



    with gr.Accordion("Advanced Options", open=False):
        with gr.Row():
            with gr.Column():
                #inversion
                src_prompt = gr.Textbox(lines=1, label="Source Prompt", interactive=True, placeholder="describe the original image")
                steps = gr.Number(value=100, precision=0, label="Num Diffusion Steps", interactive=True)
                cfg_scale_src = gr.Slider(minimum=1, maximum=15, value=3.5, label=f"Source Guidance Scale", interactive=True)
            with gr.Column():
                # reconstruction
                skip = gr.Slider(minimum=0, maximum=40, value=36, precision=0, label="Skip Steps", interactive=True)
                cfg_scale_tar = gr.Slider(minimum=7, maximum=18,value=15, label=f"Target Guidance Scale", interactive=True)
                seed = gr.Number(value=0, precision=0, label="Seed", interactive=True)
                randomize_seed = gr.Checkbox(label='Randomize seed', value=True)
            

    edit_button.click(
        fn=edit,
        inputs=[input_image,
                do_inversion, 
            src_prompt, 
            tar_prompt,
            steps,
            cfg_scale_src,
            cfg_scale_tar,
            skip,
            seed,
            randomize_seed
        ],
        outputs=[output_image, wts, zs, do_inversion],
    )

    input_image.change(
        fn = reset_do_inversion,
        outputs = [do_inversion]
    )

    src_prompt.change(
        fn = reset_do_inversion,
        outputs = [do_inversion]
    )

    # skip.change(
    #     fn = reset_latents
    # )


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
    )



demo.queue()
demo.launch(share=False)