#!/usr/bin/env python3
"""
This module defines a function for answering questions from a corpus of documents.
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np
import os


def question_answer(corpus_path):
    """
    Answers questions from multiple reference texts in a corpus.

    Args:
        corpus_path (str): The path to the corpus of reference documents.
    """
    # Load the BERT QA model from TensorFlow Hub
    qa_model_handle = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"
    try:
        qa_model = hub.load(qa_model_handle)
        print(f"BERT QA model loaded from TF Hub: {qa_model_handle}")
    except Exception as e:
        print(f"Error loading model from TF Hub '{qa_model_handle}': {e}")
        print(
            "Error loading the QA model.  This might be due to a network"
            " issue or an outdated TF Hub URL.  Please check your connection"
            " and the TF Hub for the latest model URL."
        )
        return None

    # Load the tokenizer specified
    tokenizer_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    try:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Error loading tokenizer '{tokenizer_name}': {e}")
        return None

    def get_answer(question, reference):
        """
        Helper function to get an answer for a given question and reference.
        """
        try:
            encoded_inputs = tokenizer.encode_plus(
                question,
                reference,
                add_special_tokens=True,
                max_length=512,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='tf'
            )
        except Exception as e:
            print(f"Error during tokenization: {e}")
            return None

        # Prepare model inputs
        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']
        token_type_ids = encoded_inputs['token_type_ids']

        try:
            # The model expects a dictionary as input
            outputs = qa_model({
                'input_word_ids': input_ids,
                'input_mask': attention_mask,
                'input_type_ids': token_type_ids
            })
            start_logits_pred = outputs[0]
            end_logits_pred = outputs[1]
        except Exception as e:
            print(f"Error during model prediction: {e}")
            return None

        # Find the start and end indices of the answer
        start_index = tf.argmax(start_logits_pred, axis=1).numpy()[0]
        end_index = tf.argmax(end_logits_pred, axis=1).numpy()[0]

        # --- Post-processing ---
        if start_index > end_index:
            return None

        input_ids_list = input_ids.numpy().flatten().tolist()
        try:
            sep_index = input_ids_list.index(tokenizer.sep_token_id)
            context_start_index = sep_index + 1
            if start_index < context_start_index:
                return None
        except ValueError:
            return None

        if start_index == 0 and end_index == 0:
            return None

        # Convert token indices to text
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids_list)
        answer_tokens = all_tokens[start_index:end_index + 1]
        answer_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens), skip_special_tokens=True).strip()

        return answer_text if answer_text else None


    # Read documents from the corpus path
    corpus_documents = []
    try:
        if os.path.isdir(corpus_path):
            for filename in os.listdir(corpus_path):
                filepath = os.path.join(corpus_path, filename)
                if os.path.isfile(filepath):
                    with open(filepath, "r", encoding="utf-8") as f:
                        corpus_documents.append(f.read())
        else:  # Assume it's a single file
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus_documents.append(f.read())

        if not corpus_documents:
            print(f"No documents found in corpus path: {corpus_path}")
            return None

    except Exception as e:
        print(f"Error reading corpus documents: {e}")
        return None

    while True:
        user_input = input("Q: ")
        if user_input.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break

        for doc in corpus_documents:
            answer = get_answer(user_input, doc)
            if answer:
                print(f"A: {answer}")
                break
        else:
            print("A: Sorry, I do not understand your question.")



if __name__ == '__main__':
    # Create a dummy corpus for testing
    corpus_path = "dummy_corpus"
    os.makedirs(corpus_path, exist_ok=True)
    with open(os.path.join(corpus_path, "doc1.txt"), "w", encoding="utf-8") as f:
        f.write("This is the first document. It talks about cats.")
    with open(os.path.join(corpus_path, "doc2.txt"), "w", encoding="utf-8") as f:
        f.write("The second document discusses dogs and their habits.")
    with open(os.path.join(corpus_path, "doc3.txt"), "w", encoding="utf-8") as f:
        f.write("Document three mentions both cats and dogs.")

    question_answer(corpus_path)
    import shutil
    shutil.rmtree(corpus_path)
