import streamlit as st
from peft import PeftModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the model and tokenizer (cached to avoid reloading on every interaction)
@st.cache_resource
def load_model():
    # Load the base GPT-2 model
    base_model = GPT2LMHeadModel.from_pretrained('gpt2')
    # Load the LoRA adapters and apply them to the base model
    model = PeftModel.from_pretrained(base_model, "./math_meme_model_lora")
    # Load the tokenizer saved during training
    tokenizer = GPT2Tokenizer.from_pretrained("./math_meme_model_lora")
    # Set the pad token to the EOS token (consistent with training)
    tokenizer.pad_token = tokenizer.eos_token
    # Determine device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Set model to evaluation mode for inference
    model.eval()
    return model, tokenizer, device

# Load model, tokenizer, and device once at startup
model, tokenizer, device = load_model()

# Function to generate corrections for incorrect math statements
def correct_math(incorrect_statement):
    # Construct the prompt
    prompt = f"Incorrect: {incorrect_statement}\nCorrect:"
    # Tokenize the input
    inputs = tokenizer.encode_plus(
        prompt,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    # Generate correction without computing gradients
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=5,  # Beam search for coherence
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,  # Enable sampling for diversity
            top_k=50,  # Top-k sampling
            top_p=0.95,  # Nucleus sampling
        )
    # Decode the generated text and extract the correction
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    correction = generated_text.split('Correct:')[1].strip()
    return correction

# Streamlit user interface
st.title("Math Meme Repair")
st.write("Enter an incorrect math statement to get the correct explanation.")

# Text input for the incorrect statement
incorrect_statement = st.text_input("Incorrect Math Statement:")

# Button to trigger correction
if st.button("Correct"):
    if incorrect_statement:
        # Generate and display the correction
        correction = correct_math(incorrect_statement)
        st.write("**Correction:**", correction)
    else:
        st.write("Please enter an incorrect math statement.")
