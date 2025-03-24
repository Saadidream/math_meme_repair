import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel

# Page configuration
st.set_page_config(
    page_title="Math Meme Repair",
    page_icon="🧮",
    layout="centered"
)

# App title and description
st.title("Math Meme Repair")
st.markdown("Enter an incorrect math statement to get it corrected.")

# Initialize session state for model and tokenizer
if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
    with st.spinner("Loading model... This may take a moment."):
        @st.cache_resource
        def load_model():
            tokenizer = GPT2Tokenizer.from_pretrained("./math_meme_model_lora")
            base_model = GPT2LMHeadModel.from_pretrained("gpt2")
            model = PeftModel.from_pretrained(base_model, "./math_meme_model_lora")
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                model = model.cuda()
            
            model.eval()
            return tokenizer, model
        
        st.session_state.tokenizer, st.session_state.model = load_model()

# Function to generate corrections
def generate_correction(incorrect_statement):
    tokenizer = st.session_state.tokenizer
    model = st.session_state.model
    
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

# Example selector
example_statements = {
    "Select an example (optional)": "",
    "8 ÷ 2(2+2) = 1": "8 ÷ 2(2+2) = 1",
    "5² = 10": "5² = 10",
    "1/2 + 1/2 = 1/4": "1/2 + 1/2 = 1/4",
    "Area of a circle, r=2, is π*2 = 6.28": "Area of a circle, r=2, is π*2 = 6.28",
    "x² = 4, so x = 2": "x² = 4, so x = 2"
}

selected_example = st.selectbox("Choose an example or write your own below:", 
                               options=list(example_statements.keys()))

# Input area
if selected_example != "Select an example (optional)":
    incorrect_statement = example_statements[selected_example]
else:
    incorrect_statement = ""

user_input = st.text_area("Incorrect Math Statement:", 
                         value=incorrect_statement,
                         height=100,
                         placeholder="Enter a math statement that needs correction...")

# Submit button
if st.button("Correct This Math Statement", type="primary"):
    if user_input:
        with st.spinner("Generating correction..."):
            correction = generate_correction(user_input)
        
        # Display results
        st.subheader("Corrected Statement:")
        st.success(correction)
        
        # Show original for comparison
        with st.expander("View original incorrect statement"):
            st.write(user_input)
    else:
        st.warning("Please enter a math statement to correct.")

# Sidebar with additional information
with st.sidebar:
    st.header("About Math Meme Repair")
    st.write("""
    This app uses a fine-tuned GPT-2 model to correct common mathematical 
    misconceptions often found in internet memes.
    
    The model was trained on examples of incorrect mathematical statements 
    paired with their correct explanations.
    
    **Common math misconceptions the model can correct:**
    - Order of operations (PEMDAS) errors
    - Exponent calculation mistakes
    - Fraction addition errors
    - Distribution property misapplications
    - And many more!
    """)
    
    st.write("---")
    st.write("Made with ❤️ using Streamlit and 🤗 Transformers")
