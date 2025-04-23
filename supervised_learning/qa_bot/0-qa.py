#!/usr/bin/env python3
"""
Module for performing Question Answering using a pre-trained BERT model.
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np  # Although not directly used, TF might rely on it


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question
    using the bert-uncased-tf2-qa model from TensorFlow Hub
    and the BertTokenizer, bert-large-uncased-whole-word-masking-finetuned-squad,
    from the transformers library.

    Args:
        question (str): The question to answer.
        reference (str): The reference document (context) to search for the
                         answer.

    Returns:
        str: The extracted answer snippet from the reference document, or None
             if no suitable answer is found.
    """
    # Load the tokenizer specified
    tokenizer_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    try:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Error loading tokenizer '{tokenizer_name}': {e}")
        return None

    # Load the BERT QA model from TensorFlow Hub
    qa_model_handle = "https://tfhub.dev/tensorflow/bert_uncased_tf2_qa/2"  # Corrected model URL
    try:
        qa_model = hub.load(qa_model_handle)
        print(f"BERT QA model loaded from TF Hub: {qa_model_handle}")
    except Exception as e:
        print(f"Error loading model from TF Hub '{qa_model_handle}': {e}")
        return None

    # Tokenize the input question and reference text
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

    # Get model predictions
    try:
        outputs = qa_model([input_ids, attention_mask, token_type_ids])
        start_logits_pred = outputs['start_logits']
        end_logits_pred = outputs['end_logits']
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
    answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens), skip_special_tokens=True).strip()

    return answer if answer else None



if __name__ == '__main__':
    reference_text = """
    Peer Learning Days (PLDs) are typically held on the third Tuesday of each month.
    These sessions aim to encourage collaboration and the sharing of knowledge among team members.
    The upcoming PLD is scheduled for October 22nd, 2025.
    Participants are encouraged to come prepared with topics they wish to discuss or present.
    """
    question1 = "When are PLDs held?"
    answer1 = question_answer(question1, reference_text)
    print(f"Question: {question1}\nAnswer: {answer1}\n")

    question2 = "What is the purpose of PLDs?"
    answer2 = question_answer(question2, reference_text)
    print(f"Question: {question2}\nAnswer: {answer2}\n")

    question3 = "When is the next PLD scheduled?"
    answer3 = question_answer(question3, reference_text)
    print(f"Question: {question3}\nAnswer: {answer3}\n")

    question4 = "What should participants bring?"
    answer4 = question_answer(question4, reference_text)
    print(f"Question: {question4}\nAnswer: {answer4}\n")
