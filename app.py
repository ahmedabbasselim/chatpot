from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from flask import Flask
from flask_cors import CORS
from flask import request
import json

# Load model (download on first run and reference local installation for consequent runs)
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

app = Flask(__name__)
CORS(app)

@app.route('/chatbot', methods=['Post'])
def chatbot():
    #Extract input_test from the HTTP reauest
    request_string= request.get_data()
    request_dictionary = json.loads(request_string)
    input_text = request_dictionary['prompt']
    #print("prompt")
    #print(input_text)

    # Create conversation history string
    history = "\n".join(conversation_history)
    #print("history")
    #print(history)

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history, input_text, return_tensors="pt")
    #print("input tokens")
    #print(input)
    # Generate the response from the model
    outputs = model.generate(**inputs)
    #print("LLM output")
    #print(outputs)
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    #print('response')
    #print(response)

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)
    #print("conversation_history")    
    #print(conversation_history)
    return response

if __name__ == '__main__':
    app.run()
