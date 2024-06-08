import torch  
import time
from meshgpt_pytorch import ( 
    MeshTransformer,
    mesh_render
)
import igl

import gradio as gr
import numpy as np 

transformer = MeshTransformer.from_pretrained("MarcusLoren/MeshGPT-preview")
def save_as_obj(file_path):
    v, f = igl.read_triangle_mesh(file_path)
    v, f, _, _ = igl.remove_unreferenced(v, f)
    c, _ = igl.orientable_patches(f)
    f, _ = igl.orient_outward(v, f, c)
    igl.write_triangle_mesh(file_path, v, f)
    return file_path

def predict(text, num_input, num_temp):
    transformer.eval()
    labels = [label.strip() for label in text.split(',')] 
    output = []
    current_time = time.time() 
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
    print(formatted_time, " Input:", text, "num_input", num_input, "num_temp",num_temp)
    if num_input > 1:
        for label in labels:
            output.append((transformer.generate(texts = [label ] * num_input, temperature = num_temp)))   
    else:
        output.append((transformer.generate(texts = labels  , temperature = num_temp)))   
            
    mesh_render.save_rendering('./render.obj', output) 
    return save_as_obj('./render.obj')

gradio_app = gr.Interface(
    predict, 
    inputs=[
        gr.Textbox(label="Enter labels, separated by commas"),
        gr.Number(value=1, label="Number of examples per input"),
        gr.Slider(minimum=0, maximum=1, value=0, label="Temperature (0 to 1)")
    ],
    outputs=gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Model"),
    title="MeshGPT Inference - (Rendering doesn't work, please download for best result)",
)

if __name__ == "__main__":
    gradio_app.launch(share=False)