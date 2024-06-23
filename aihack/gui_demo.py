import gradio as gr
from guard import Guard
from modules import GPT

gpt = GPT()
safe_llm = Guard(gpt)

with gr.Blocks() as demo:
    prompt = gr.Textbox(label="Prompt")
    with gr.Row():
        baseline = gr.Textbox(label="Output")
        with gr.Column():
            checkbox_options = ["classify", "sanitize"]
            flags = gr.CheckboxGroup(choices=checkbox_options, label="Flags")
            classification = gr.Textbox(label="Classification")
            sanitized = gr.Textbox(label="Sanitized")
            clean = gr.Textbox(label="Output")
    submit_btn = gr.Button("Submit")

    @submit_btn.click(inputs=[prompt, flags], outputs=[baseline, clean, classification, sanitized])
    def run_models(inputs, flags):
        classify = 'classify' in flags
        sanitize = 'sanitize' in flags
        outs = safe_llm(inputs, classifier=classify, sanitizer=sanitize)
        clean = outs[0]
        classification = outs[1]['class'][0]
        sanitized = outs[1]['sanitized'][0]
        return gpt.forward(inputs), clean, classification, sanitized

demo.launch()