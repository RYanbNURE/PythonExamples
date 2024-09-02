import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the model and tokenizer you want to use
model_name = "EleutherAI/gpt-neo-2.7B"  # Example with GPT-Neo model

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Check for GPU availability and move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Function to generate text
def generate_text(prompt, max_length=50):
    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    inputs = inputs.to(device)

    # Generate text using the model
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# Test the function
prompt = "Once upon a time"
generated_text = generate_text(prompt, max_length=50)
print(f"Prompt: {prompt}\nGenerated Text: {generated_text}")