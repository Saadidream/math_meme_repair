import gradio as gr
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel

# Load the tokenizer and model
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("./math_meme_model_lora")
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = PeftModel.from_pretrained(base_model, "./math_meme_model_lora")
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()
    return tokenizer, model

# Generate correction for a given incorrect math statement
def generate_correction(incorrect_statement, tokenizer, model):
    prompt = f"Incorrect: {incorrect_statement}\nCorrect:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate output
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=150,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract just the correction part
    try:
        correction = generated_text.split("Correct:")[1].strip()
    except IndexError:
        correction = generated_text
    
    return correction

# Main interface function
def correct_math_meme(incorrect_statement):
    tokenizer, model = load_model()
    correction = generate_correction(incorrect_statement, tokenizer, model)
    return correction

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Math Meme Repair") as demo:
        gr.Markdown("# Math Meme Repair")
        gr.Markdown("Enter an incorrect math statement to get it corrected.")
        
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Incorrect Math Statement",
                    placeholder="Example: 8 ÷ 2(2+2) = 1",
                    lines=3
                )
                submit_btn = gr.Button("Generate Correction", variant="primary")
            
            with gr.Column():
                output_text = gr.Textbox(
                    label="Corrected Math Statement",
                    lines=6
                )
        
        # Examples to show how the app works
        examples = [
            ["8 ÷ 2(2+2) = 1"],
            ["5² = 10"],
            ["1/2 + 1/2 = 1/4"],
            ["Area of a circle, r=2, is π*2 = 6.28"],
            ["x² = 4, so x = 2"]
        ]
        gr.Examples(examples, input_text)
        
        submit_btn.click(fn=correct_math_meme, inputs=input_text, outputs=output_text)
    
    return demo

# Main entry point
if __name__ == "__main__":
    print("Loading Math Meme Repair app...")
    demo = create_interface()
    demo.launch(share=True)  # Set share=False if you don't want a public URL
    print("Math Meme Repair app is running!")
