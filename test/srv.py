from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'JetBrains/deepseek-coder-1.3B-kexer'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')

app = Flask(__name__)

# Define your model and prompt here (or load them)
# model = load_model_function()  # Replace with your actual model loading code

@app.route('/api', methods=['POST'])
def generate_response():
    # Get the prompt from the incoming JSON request
    data = request.get_json()
    prompt = data.get('prompt', '')

    # Generate response based on the prompt using your model (example)
    # response = model.generate(prompt)  # Replace with your actual model inference code
    input_ids = tokenizer.encode(
                prompt, return_tensors='pt'
                ).to('cuda')

    # Generate
    output = model.generate(
                input_ids, max_length=60, num_return_sequences=1,
                    early_stopping=True, pad_token_id=tokenizer.eos_token_id,
                    )

    # Decode output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    #print(generated_text)



    # For now, let's return the prompt as a placeholder response
    response = f"Received prompt: {generated_text}"

    return jsonify({"response": generated_text})

if __name__ == '__main__':
    app.run(debug=True)

