from queryHandler import get_relevant_texts
from transformers import BartTokenizer, BartForConditionalGeneration

while __name__ == "__main__":
 user_input = input("Please enter your query: ")
 if user_input.lower() == 'exit':
            break
    
   
 answer = get_relevant_texts(user_input)
 def generate_response(query, relevant_texts):
    context = "\n\n".join(relevant_texts)
    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
    model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')

    # Tokenize inputs and generate summary
    inputs = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs.input_ids, num_beams=4, min_length=30, max_length=150, early_stopping=True)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output.strip()


 response=generate_response(user_input,answer)
 print(response)

