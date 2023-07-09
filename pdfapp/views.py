import PyPDF2
import pdfplumber
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForSequenceClassification
from django.shortcuts import render, redirect
import os

def home(request):
    return render(request, 'pdfapp/home.html')

def upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['pdf_file']
        pdf_path = handle_uploaded_file(uploaded_file)  # Save the uploaded file and get its path
        return redirect('summarize', pdf_path=pdf_path)

    return render(request, 'pdfapp/upload.html')

def summarize(request, pdf_path):
    # Parse PDF and extract text content
    with pdfplumber.open(pdf_path) as pdf:
        text_content = ""
        for page in pdf.pages:
            text_content += page.extract_text()

    # Preprocessing steps
    text_content = text_content.replace("\n", " ")  # Remove line breaks
    text_content = re.sub(r"\s+", " ", text_content)  # Replace multiple spaces with a single space
    text_content = text_content.lower()  # Convert to lowercase

    # Tokenize the preprocessed text into sentences
    sentences = sent_tokenize(text_content)

    # Initialize the summarization model and tokenizer
    model_name = 't5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Initialize the document classification model and tokenizer
    classification_model_name = 'bert-base-uncased'
    classification_tokenizer = BertTokenizer.from_pretrained(classification_model_name)
    classification_model = BertForSequenceClassification.from_pretrained(classification_model_name, num_labels=2)

    # List to store the chunked data and summaries
    chunked_data = []
    summaries = []

    # Chunk the preprocessed data and store in the list
    current_chunk = ""
    for sentence in sentences:
        words = word_tokenize(sentence)
        if len(current_chunk) + len(words) <= 950:
            current_chunk += " " + sentence
        else:
            chunked_data.append(current_chunk.strip())
            current_chunk = sentence

    # Add the last chunk if it's not empty
    if current_chunk:
        chunked_data.append(current_chunk.strip())

    # Generate summary for each chunk
    for i, chunk in enumerate(chunked_data):
        # Tokenize the chunk
        inputs = tokenizer.encode_plus(chunk, truncation=True, padding='longest', max_length=1024, return_tensors='pt')

        # Generate the summary
        summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length=150, early_stopping=True)
        summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

        summaries.append(summary)

    # Concatenate the summaries
    concatenated_summary = ' '.join(summaries)

    # Post-processing to make the summary more precise
    def remove_duplicates(text):
        # Split the text into sentences
        sentences = text.split('. ')

        # Remove duplicate sentences
        unique_sentences = []
        for sentence in sentences:
            if sentence not in unique_sentences:
                unique_sentences.append(sentence)

        # Join the unique sentences back into a single string
        processed_text = '. '.join(unique_sentences)

        return processed_text

    def post_process_text(text):
        # Remove UTC timestamps
        text = re.sub(r'utc \d{2}:\d{2}:\d{2}', '', text, flags=re.IGNORECASE)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    # Remove duplicate sentences
    processed_summary = remove_duplicates(concatenated_summary)

    # Perform post-processing
    final_summary = post_process_text(processed_summary)


    # Classify the final summary using BERT
    classification_inputs = classification_tokenizer.encode_plus(final_summary, truncation=True, padding='longest', max_length=512, return_tensors='pt')
    logits = classification_model(**classification_inputs)[0]
    predicted_label_id = logits.argmax().item()
    document_type = 'Legal' if predicted_label_id == 1 else 'Business'  # Assuming label 1 corresponds to Legal and label 0 corresponds to Business

    return render(request, 'pdfapp/summary.html', {'final_summary': final_summary, 'document_type': document_type})

def handle_uploaded_file(uploaded_file):
    # Get the file name
    file_name = uploaded_file.name

    # Construct the file path where the file will be saved
    file_path = os.path.join('media', file_name)

    # Save the uploaded file
    with open(file_path, 'wb') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)

    return file_path