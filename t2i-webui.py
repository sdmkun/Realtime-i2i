# サードパーティライブラリ
import webuiapi
from PIL import Image
# WebUIのホスト、ポート番号
IMAGE_SERVER_HOST = "127.0.0.1"
IMAGE_SERVER_PORT = 7860
# プロンプト
PROMPT = """{prompt} <lora:LCM_LoRA_Weights_SD15:1>"""
# NEGATIVE_PROMPT = """EasyNegativeV2"""
NEGATIVE_PROMPT = """"""
# ControlNetに入力する画像
# MEDIA_PIPE_FACE_IMAGE = "./img/base_face.png"
MEDIA_PIPE_POSE_IMAGE = "./img/base_pose.png"
MEDIA_PIPE_REFERENCE_IMAGE = "./img/base_reference.png"
# 生成画像時の設定
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
BATCH_SIZE = 1
image_api = webuiapi.WebUIApi(IMAGE_SERVER_HOST, IMAGE_SERVER_PORT, sampler='LCM')
# モデル名を取得
model_name = None
for controlnet_model in image_api.controlnet_model_list():
    if "openpose" in controlnet_model:
        model_name = controlnet_model
# ControlNet用のオブジェクトの作成
# base_face_img = Image.open(MEDIA_PIPE_FACE_IMAGE)
base_pose_img = Image.open(MEDIA_PIPE_POSE_IMAGE)
unit_base_openpose = webuiapi.ControlNetUnit(
    input_image=base_pose_img,
    processor_res=512,
    module='openpose',
    model=model_name
)
base_reference_img = Image.open(MEDIA_PIPE_REFERENCE_IMAGE)
unit_base_reference = webuiapi.ControlNetUnit(
    input_image=base_reference_img,
    processor_res=512, # 最初から解像度を揃えておけば速い？
    module='reference_only',
    control_mode=0
    # model=model_name
)
# control_net_unit = [unit_base_openpose, unit_base_reference]
control_net_unit = [unit_base_openpose]
# プロンプトに主題を設定して画像を生成
prompt = PROMPT.format(prompt="1man, a character from overwatch with chin beard, with a black outfit and brown hair, Dr. Atl, overwatch, concept art, rayonism")
result = image_api.txt2img(
    prompt=prompt,
    negative_prompt=NEGATIVE_PROMPT,
    width=IMAGE_WIDTH,
    height=IMAGE_HEIGHT,
    # seed=1,
    cfg_scale=1,
    steps=4,
    n_iter=10,
    batch_size=BATCH_SIZE,
    controlnet_units=control_net_unit,
    do_not_save_samples=True,
    do_not_save_grid=True,
)
for i, image in enumerate(result.images):
    # if i != len(result.images) - 1:
    image.save(f"./base_image{i}.png")
