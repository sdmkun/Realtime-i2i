

import cv2
import numpy as np

from diffusers import AutoPipelineForImage2Image, UNet2DConditionModel, LCMScheduler
from diffusers.utils import load_image
import torch
from PIL import Image


def get_pipe(config):
    # pipe = AutoPipelineForImage2Image.from_single_file(
        # config.generation_model_name.get()
    pipe = AutoPipelineForImage2Image.from_pretrained(
        config.generation_model_name.get(), torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")

    pipe.load_lora_weights(config.lcm_model_name.get())
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    # pipe.enable_model_cpu_offload()

    if pipe.safety_checker is not None:
        pipe.safety_checker = lambda images, **kwargs: (images, [False])

    return pipe


def LCM_run(config, pipe):
    cv2.namedWindow("Window Capture", cv2.WINDOW_NORMAL)
    config.running = True
    while config.running:
        # カメラ画像を取得
        frame = config.screen_capture.capture()
        img_np = np.array(frame)
        #img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_np)

        generator = torch.Generator("cuda").manual_seed(2500)

        img = pipe(
            strength=config.strength_value,
            prompt=config.prompt.get(),
            image=img,
            width=256,
            height=256,
            num_inference_steps=config.num_inference_steps_value,
            guidance_scale=1,
            generator=generator
        ).images[0]

        # PILイメージをnumpy配列に変換
        img = np.array(img)

        # OpenCVでは色の順番がBGRなので、RGBからBGRに変換
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # キャプチャした画像を同じウィンドウで更新して表示
        cv2.imshow("Window Capture", img)

        # 'q'を押したらループを抜ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    config.close()
    cv2.destroyAllWindows()

