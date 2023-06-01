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
  wt, zs, wts = inversion_forward_process(sd_pipe, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src, prog_bar=True, num_inference_steps=num_diffusion_steps)
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
            'Examples/gnochi_mirror_reconstrcution.png', 
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
            seed = 0,
            left = 0,
            right = 0,
            top = 0,
            bottom = 0
):
    torch.manual_seed(seed)
     # offsets=(0,0,0,0)
    x0 = load_512(input_image, left,right, top, bottom, device)


    # invert and retrieve noise maps and latent
    wt, zs, wts = invert(x0 =x0 , prompt_src=src_prompt, num_diffusion_steps=steps, cfg_scale_src=cfg_scale_src)

    #
    xT=wts[skip]
    etas=1.0
    prompts=[tar_prompt]
    cfg_scales=[cfg_scale_tar]
    prog_bar=False
    zs=zs[skip:]


    batch_size = len(prompts)

    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1,1,1,1).to(sd_pipe.device)

    text_embeddings = encode_text(model, prompts)
    uncond_embedding = encode_text(model, [""] * batch_size)

    if etas is None: etas = 0
    if type(etas) in [int, float]: etas = [etas]*sd_pipe.scheduler.num_inference_steps
    assert len(etas) == sd_pipe.scheduler.num_inference_steps
    timesteps = sd_pipe.scheduler.timesteps.to(sd_pipe.device)

    xt = xT.expand(batch_size, -1, -1, -1)
    op = tqdm(timesteps[-zs.shape[0]:]) if prog_bar else timesteps[-zs.shape[0]:] 

    t_to_idx = {int(v):k for k,v in enumerate(timesteps[-zs.shape[0]:])}

    for t in op:
        idx = t_to_idx[int(t)]        
        ## Unconditional embedding
        with torch.no_grad():
            uncond_out = sd_pipe.unet.forward(xt, timestep =  t, 
                                            encoder_hidden_states = uncond_embedding)

            ## Conditional embedding  
        if prompts:  
            with torch.no_grad():
                cond_out = sd_pipe.unet.forward(xt, timestep =  t, 
                                                encoder_hidden_states = text_embeddings)
            
        
        z = zs[idx] if not zs is None else None
        z = z.expand(batch_size, -1, -1, -1)
        if prompts:
            ## classifier free guidance
            noise_pred = uncond_out.sample + cfg_scales_tensor * (cond_out.sample - uncond_out.sample)
        else: 
            noise_pred = uncond_out.sample
        # 2. compute less noisy image and set x_t -> x_t-1  
        xt = reverse_step(sd_pipe, noise_pred, t, xt, eta = etas[idx], variance_noise = z)

        # interm denoised img
        with autocast("cuda"), inference_mode():
            x0_dec = sd_pipe.vae.decode(1 / 0.18215 * xt).sample
            if x0_dec.dim()<4:
                x0_dec = x0_dec[None,:,:,:]
                interm_img = image_grid(x0_dec)
                yield interm_img
      
    yield interm_img
    
    # # vae decode image
    # with autocast("cuda"), inference_mode():
    #     x0_dec = sd_pipe.vae.decode(1 / 0.18215 * w0).sample
    # if x0_dec.dim()<4:
    #     x0_dec = x0_dec[None,:,:,:]
    # img = image_grid(x0_dec)
    # return img

    # output = sample(wt, zs, wts, prompt_tar=tar_prompt)

    # return output





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
    with gr.Row():
        src_prompt = gr.Textbox(lines=1, label="Source Prompt", interactive=True, placeholder="optional: describe the original image")
        tar_prompt = gr.Textbox(lines=1, label="Target Prompt", interactive=True, placeholder="optional: describe the target image")

    with gr.Row():
        input_image = gr.Image(label="Input Image", interactive=True)
        input_image.style(height=512, width=512)
        inverted_image = gr.Image(label=f"Reconstructed Image", interactive=False)
        inverted_image.style(height=512, width=512)
        output_image = gr.Image(label=f"Edited Image", interactive=False)
        output_image.style(height=512, width=512)


    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            invert_button = gr.Button("Invert")
        with gr.Column(scale=1, min_width=100):
            edit_button = gr.Button("Edit")


    with gr.Accordion("Advanced Options", open=False):
        with gr.Row():
            with gr.Column():
                #inversion
                steps = gr.Number(value=100, precision=0, label="Num Diffusion Steps", interactive=True)
                cfg_scale_src = gr.Slider(minimum=1, maximum=15, value=3.5, label=f"Source Guidance Scale", interactive=True)
      
                # reconstruction
                skip = gr.Slider(minimum=0, maximum=40, value=36, precision=0, label="Skip Steps", interactive=True)
                cfg_scale_tar = gr.Slider(minimum=7, maximum=18,value=15, label=f"Target Guidance Scale", interactive=True)
                seed = gr.Number(value=0, precision=0, label="Seed", interactive=True)

            #shift
            with gr.Column():
                left = gr.Number(value=0, precision=0, label="Left Shift", interactive=True)
                right = gr.Number(value=0, precision=0, label="Right Shift", interactive=True)
                top = gr.Number(value=0, precision=0, label="Top Shift", interactive=True)
                bottom = gr.Number(value=0, precision=0, label="Bottom Shift", interactive=True)

            
          

    # gr.Markdown(help_text)

    invert_button.click(
        fn=edit,
        inputs=[input_image, 
                    src_prompt, 
                    src_prompt,
                    steps,
                    cfg_scale_src,
                    cfg_scale_tar,
                    skip,
                    seed,
                    left,
                    right,
                    top,
                    bottom
        ],
        outputs = [inverted_image],
    )

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
            left,
            right,
            top,
            bottom
        ],
        outputs=[output_image],
    )


    gr.Examples(
        label='Examples', 
        examples=get_example(), 
        inputs=[input_image, src_prompt, tar_prompt, steps,
                    cfg_scale_tar,
                    skip,
                    cfg_scale_tar,
                    inverted_image, output_image
               ],
        outputs=[inverted_image,output_image ],
        # fn=edit,
        # cache_examples=True
    )



demo.queue()
demo.launch(share=False)