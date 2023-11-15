import transformers
import json

from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

# https://huggingface.co/facebook/bart-large-cnn
def simplify_caption(caption, model_name="facebook/bart-large-cnn"):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the model
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenize the input caption
    inputs = tokenizer.encode("simplify: " + caption, return_tensors="tf", max_length=512, truncation=True)

    # Generate the simplified output
    outputs = model.generate(inputs, max_length=70, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the generated tokens to a string
    simplified_caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return simplified_caption

input_file = open ('artifact_images_w_sentence_captions.json')
json_array = json.load(input_file)

for item in json_array:
    caption = item.get('caption', '')  # Replace 'caption' with the actual key for captions in your JSON
    item['simplified_caption'] = simplify_caption(caption)

# Write the updated data to a new file
output_file = open('updated_artifact_images.json', 'w')
json.dump(json_array, output_file)
output_file.close()
