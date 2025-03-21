import streamlit as st
from peft import PeftModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the model and tokenizer
@st.cache_resource
def load_model():
    base_model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = PeftModel.from_pretrained(base_model, "./math_meme_model_lora")
    tokenizer = GPT2Tokenizer.from_pretrained("./math_meme_model_lora")
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_model()

# Function to generate corrections
def correct_math(incorrect_statement):
    prompt = f"Incorrect: {incorrect_statement}\nCorrect:"
    inputs = tokenizer.encode_plus(prompt, return_tensors='pt', padding=True, truncation=True).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=5,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    correction = generated_text.split('Correct:')[1].strip()
    return correction

# Streamlit UI
st.title("Math Meme Repair")
st.write("Enter an incorrect math statement to get the correct explanation.")
incorrect_statement = st.text_input("Incorrect Math Statement:")
if st.button("Correct"):
    if incorrect_statement:
        correction = correct_math(incorrect_statement)
        st.write("**Correction:**", correction)
    else:
        st.write("Please enter an incorrect math statement.")
