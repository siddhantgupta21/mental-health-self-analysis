import streamlit as st
import numpy as np
import joblib  # For Gemma 2B API
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login


# Mapping of classes to descriptions
conditions = {
    0: "Minimal Depression",
    1: "Mild Depression",
    2: "Moderate Depression",
    3: "No Depression",
    4: "Severe Depression"
}

# Function to get LLM explanation from Gemma
def get_explanation(predicted_class):
  device = "cuda" if torch.cuda.is_available() else "cpu"

# Log in to Hugging Face Hub
  login(token="")#Enter your gemma access key

# Load the tokenizer and model
  tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
  model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it").to(device)
  condition = conditions.get(predicted_class, "unknown condition")

    # Define the prompt
  prompt = f"""
  You are a mental health expert. Explain what it means to have {condition} in simple and empathetic terms.
  Provide 3 actionable coping mechanisms and 2 potential next steps for someone with {condition}.
  Write the response in a professional yet compassionate tone.

  """

  # Tokenize the prompt and move inputs to the correct device
  inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

  # Generate the output
  outputs = model.generate(
      inputs["input_ids"],
      max_length=512,
      num_beams=5,
      early_stopping=True,
      temperature=0.7,
      no_repeat_ngram_size=2
  )

  # Decode the output
  explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return explanation

    

# Streamlit UI
st.title("Mental Health Self-Analysis Chatbot")

st.write("Enter your GAD, PHQ, and Epworth scores below:")

# User inputs
gad_score = st.number_input("GAD Score (0-21)", min_value=0, max_value=21, step=1)
phq_score = st.number_input("PHQ Score (0-27)", min_value=0, max_value=27, step=1)
epworth_score = st.number_input("Epworth Score (0-24)", min_value=0, max_value=24, step=1)

if st.button("Analyze"):
    # Make a prediction
    model = joblib.load('mental_health_model.pkl')
    input_data = [[gad_score, phq_score, epworth_score]]
    predicted_class=model.predict(input_data)[0]

    
    # Get explanation from LLM
    explanation = get_explanation(predicted_class)
    
    # Display results
    st.subheader(f"Prediction: {conditions[predicted_class]}")
    st.write(explanation)
