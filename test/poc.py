from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pre-trained model and tokenizer
model_name = 'JetBrains/deepseek-coder-1.3B-kexer'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')

# Create and encode input
input_text = """\
This function takes an integer n and returns factorial of a number:
fun factorial(n: Int): Int {\
"""
input_ids = tokenizer.encode(
    input_text, return_tensors='pt'
).to('cuda')

# Generate
output = model.generate(
    input_ids, max_length=60, num_return_sequences=1, 
    early_stopping=True, pad_token_id=tokenizer.eos_token_id,
)

# Decode output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
