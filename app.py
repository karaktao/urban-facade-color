import gradio as gr
from PIL import Image
from backend.core import analyze_image_pil, to_base64_png

def pipeline(pil_img: Image.Image):
    out, colors = analyze_image_pil(pil_img)
    data_url = "data:image/png;base64," + to_base64_png(out)
    return data_url, colors


with gr.Blocks() as demo:
    gr.Markdown("## Urban Facade Color (CPU Space)\n上传街景图获得建筑立面主色（右侧色卡）")
    inp = gr.Image(type="pil", label="Upload image")
    btn = gr.Button("Analyze")
    out_img = gr.Image(label="Result w/ palette (right)")
    out_json = gr.JSON(label="Palette [RGB, ratio]")
    btn.click(pipeline, inputs=inp, outputs=[out_img, out_json])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)