# Modified from https://huggingface.co/spaces/PolyU-ChenLab/UniPixel/blob/main/app.py
import os
from pathlib import Path
import random
import re
import colorsys
from PIL import Image
import matplotlib as mpl
import numpy as np
import uuid
import imageio.v3 as iio

import torch
from torchvision.transforms.functional import to_pil_image
from huggingface_hub import hf_hub_download

import spaces
import gradio as gr

GRADIO_TMP = os.path.join(os.path.dirname(__file__), ".gradio_tmp")
Path(GRADIO_TMP).mkdir(parents=True, exist_ok=True)

os.environ["GRADIO_TEMP_DIR"] = GRADIO_TMP
os.environ["TMPDIR"] = GRADIO_TMP
os.environ["TEMP"] = GRADIO_TMP
os.environ["TMP"] = GRADIO_TMP

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config
from visualizer import sample_color, draw_mask

class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

def extract_mt_token_ids_v1(text):
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x) for x in re.findall(pattern, text)]

def extract_mt_token_ids_v2(text):
    pattern = re.compile(r'<\|mt_start\|><\|mt_(\d{4})\|><\|mt_(\d{4})\|><\|mt_end\|>')
    matches = pattern.findall(text)
    ret_list = []
    for num1, num2 in matches:
        ret_list.append(int(num1))
        ret_list.append(int(num2))
    return ret_list

def find_first_index(arr, value):
    indices = np.where(arr == value)[0]
    
    return indices[0] if len(indices) > 0 else -1

def fix_mt_format_comprehensive(text):
    pattern_too_many = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_\d+\|>)(?:<\|mt_\d+\|>)+<\|mt_end\|>'
    replacement_too_many = r'\1\2\3<|mt_end|>'
    text = re.sub(pattern_too_many, replacement_too_many, text)

    pattern_too_few_with_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_end\|>)'
    replacement_too_few = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_with_end, replacement_too_few, text)

    pattern_too_few_no_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(?!<\|mt_)'
    replacement_too_few_no_end = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_no_end, replacement_too_few_no_end, text)
    return text


MODEL = 'zhouyik/Qwen3-VL-8B-SAMTok'

TITLE = 'SAMTok: Representing Any Mask with Two Words'

HEADER = """
<p align="center" style="margin: 1em 0 2em;"><img width="260" src="https://github.com/bytedance/Sa2VA/blob/main/projects/samtok/figs/logo.png"></p>
<h3 align="center">SAMTok: Representing Any Mask with Two Words</h3>
<div style="display: flex; justify-content: center; gap: 5px;">
    <a href="https://github.com/bytedance/Sa2VA/tree/main/projects/samtok" target="_blank"><img src="https://img.shields.io/badge/arXiv-2509.18094-red"></a>
    <a href="https://github.com/bytedance/Sa2VA/tree/main/projects/samtok" target="_blank"><img src="https://img.shields.io/badge/Project-Page-brightgreen"></a>
    <a href="https://huggingface.co/collections/zhouyik/samtok" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a>
    <a href="https://github.com/bytedance/Sa2VA" target="_blank"><img src="https://img.shields.io/github/stars/bytedance/Sa2VA"></a>
</div>
<p style="margin-top: 1em;">SAMTok provides a unified mask-token interface for MLLMs. (1) SAMTok compresses region masks into two discrete tokens and faithfully reconstructs them across diverse visual domains. (2) Injecting these mask tokens into MLLMs enables a wide range of region-level mask generation and understanding tasks. (3) The text-based representation of region masks allows a purely textual answer-matching reward for the GRPO of the mask generation task.</p>
"""

JS = """
function init() {
    if (window.innerWidth >= 1536) {
        document.querySelector('main').style.maxWidth = '1536px'
    }
    document.getElementById('query_1').addEventListener('keydown', function f1(e) { if (e.key === 'Enter') { document.getElementById('submit_1').click() } })
    document.getElementById('query_2').addEventListener('keydown', function f2(e) { if (e.key === 'Enter') { document.getElementById('submit_2').click() } })
    document.getElementById('query_3').addEventListener('keydown', function f3(e) { if (e.key === 'Enter') { document.getElementById('submit_3').click() } })
    document.getElementById('query_4').addEventListener('keydown', function f4(e) { if (e.key === 'Enter') { document.getElementById('submit_4').click() } })
}
"""

device = torch.device('cuda')

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype="auto"
).cuda().eval()

processor = AutoProcessor.from_pretrained(MODEL)

# build vq-sam2 model
sam2_ckpt_local = hf_hub_download(repo_id=MODEL, filename="sam2.1_hiera_large.pt")
mask_tokenizer_local = hf_hub_download(repo_id=MODEL, filename="mask_tokenizer_256x2.pth")
CODEBOOK_SIZE = 256
CODEBOOK_DEPTH = 2
sam2_config = SAM2Config(
    ckpt_path=sam2_ckpt_local,
)
vq_sam2_config = VQ_SAM2Config(
    sam2_config=sam2_config,
    codebook_size=CODEBOOK_SIZE,
    codebook_depth=CODEBOOK_DEPTH,
    shared_codebook=False,
    latent_dim=256,
)
vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()
state = torch.load(mask_tokenizer_local, map_location="cpu")
vq_sam2.load_state_dict(state)
sam2_image_processor = DirectResize(1024)


colors = sample_color()
color_map = {f'Target {i + 1}': f'#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}' for i, c in enumerate(colors * 255)}
color_map_light = {
    f'Target {i + 1}': f'#{int(c[0] * 127.5 + 127.5):02x}{int(c[1] * 127.5 + 127.5):02x}{int(c[2] * 127.5 + 127.5):02x}'
    for i, c in enumerate(colors)
}

def enable_btns():
    return (gr.Button(interactive=True), ) * 4


def disable_btns():
    return (gr.Button(interactive=False), ) * 4


def reset_seg():
    return 16, gr.Button(interactive=False)


def reset_reg():
    return 1, gr.Button(interactive=False)

@spaces.GPU
def infer_seg(media, query):
    global model

    if not media:
        gr.Warning('Please upload an image')
        return None, None, None

    if not query:
        gr.Warning('Please provide a text prompt.')
        return None, None, None

    image = Image.open(media).convert('RGB')
    ori_width, ori_height = image.size
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": media,
                },
                {"type": "text", "text": query},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    model = model.to(device)

    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=1024,
        do_sample=False,
        top_p=1.0,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    quant_ids = extract_mt_token_ids_v1(output_text)
    if len(quant_ids) % CODEBOOK_DEPTH != 0:
        output_text = fix_mt_format_comprehensive(output_text)
        quant_ids = extract_mt_token_ids_v2(output_text)

    batch_size = len(quant_ids) // CODEBOOK_DEPTH
    remap_quant_ids = []
    tags = []
    for bs_id in range(batch_size):
        chunk_quant_ids = quant_ids[bs_id*CODEBOOK_DEPTH:(bs_id+1)*CODEBOOK_DEPTH]
        tags.append(f'<|mt_start|><|mt_{str(chunk_quant_ids[0]).zfill(4)}|><|mt_{str(chunk_quant_ids[1]).zfill(4)}|><|mt_end|>')
        remap_chunk_quant_ids = [quant_id - book_id*CODEBOOK_SIZE for book_id, quant_id in enumerate(chunk_quant_ids)]
        code1 = remap_chunk_quant_ids[0]
        code2 = remap_chunk_quant_ids[1]
        if not (code2 >= 0 and code2 < CODEBOOK_SIZE):
            code2 = -1
        remap_chunk_quant_ids_error_handle = [code1, code2]
        remap_quant_ids.append(remap_chunk_quant_ids_error_handle)

    batch_size = len(remap_quant_ids)
    sam2_image = np.array(image)
    sam2_image = sam2_image_processor.apply_image(sam2_image)
    sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
    sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
    sam2_pixel_values = sam2_pixel_values.repeat(batch_size, 1, 1, 1)

    quant_ids = torch.LongTensor(remap_quant_ids).to(vq_sam2.device)

    with torch.no_grad():
        _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
    _pred_masks = torch.nn.functional.interpolate(_pred_masks, size=(ori_height, ori_width), mode='bilinear')
    _pred_masks = _pred_masks > 0.5
    # _pred_masks = _pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)
    _pred_masks = _pred_masks.long().unsqueeze(2).cpu() # n, 1, 1, h, w 

    entities = []
    unique_tags = list(set(tags))
    entity_names = []
    for i, tag in enumerate(unique_tags):
        for m in re.finditer(re.escape(tag), output_text):
            entities.append(dict(entity=f'Target {i + 1}', start=m.start(), end=m.end()))
            entity_names.append(f'Target {i + 1}')
    
    answer = dict(text=output_text, entities=entities)

    frames = torch.from_numpy(np.array(image)).unsqueeze(0)
    imgs = draw_mask(frames, _pred_masks, colors=colors)

    path = f"/tmp/{uuid.uuid4().hex}.png"
    iio.imwrite(path, imgs, duration=100, loop=0)

    masks = media, [(m[0, 0].numpy(), entity_names[i]) for i, m in enumerate(_pred_masks)]

    return answer, masks, path


def build_demo():
    with gr.Blocks(title=TITLE, js=JS, theme=gr.themes.Soft()) as demo:
        gr.HTML(HEADER)

        # with gr.Tab('Mask Generation'):
        download_btn_1 = gr.DownloadButton(label='📦 Download', interactive=False, render=False)
        msk_1 = gr.AnnotatedImage(label='De-tokenized 2D masks', color_map=color_map, render=False)
        ans_1 = gr.HighlightedText(
            label='Model Response', color_map=color_map_light, show_inline_category=False, render=False)
        with gr.Row():
            with gr.Column():
                media_1 = gr.Image(type='filepath')

                sample_frames_1 = gr.Slider(1, 32, value=16, step=1, visible=False)

                query_1 = gr.Textbox(label='Text Prompt', placeholder='Please segment the...', elem_id='query_1')

                with gr.Row():
                    random_btn_1 = gr.Button(value='🔮 Random', visible=False)

                    reset_btn_1 = gr.ClearButton([media_1, query_1, msk_1, ans_1], value='🗑️ Reset')
                    reset_btn_1.click(reset_seg, None, [sample_frames_1, download_btn_1])

                    download_btn_1.render()

                    submit_btn_1 = gr.Button(value='🚀 Submit', variant='primary', elem_id='submit_1')
            
            with gr.Column():
                msk_1.render()
                ans_1.render()

        ctx_1 = submit_btn_1.click(disable_btns, None, [random_btn_1, reset_btn_1, download_btn_1, submit_btn_1])
        ctx_1 = ctx_1.then(infer_seg, [media_1, query_1], [ans_1, msk_1, download_btn_1])
        ctx_1.then(enable_btns, None, [random_btn_1, reset_btn_1, download_btn_1, submit_btn_1])
        # with gr.Tab('Mask Understanding'):
        #     pass

    return demo

if __name__ == '__main__':
    demo = build_demo()

    demo.queue()
    demo.launch(server_name='::')