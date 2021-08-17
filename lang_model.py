#!/usr/bin/env python
# coding: utf-8

# In[26]:


import re
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras import layers, activations, models, preprocessing
from tensorflow.keras import preprocessing, utils
import os
import yaml
import json
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd


# Opening the parsed Json File

# In[27]:


with open('Z.json') as f:
    data = json.load(f)


# Data variable currently has a dictionary of key value pairs where the key represents the patois and the value represents the english translation
# Now, we're going to split the key value pairs into their respective lists
#

# In[28]:


english = list(data.keys())
patois = list(data.values())


# In[29]:


questions_for_token = patois
answers_for_token = english
embed_size = 100  # define the vector size based on word your embedding
max_features = 6000  # to restrict your number of unique words
maxlen = 100


# Outlining anmd defining preprocessing functions

# In[30]:


def processTweet(chat):

    chat = re.sub(r'[\.!:\?\-\'\"\\/]', r'', chat)
    chat = chat.strip('\'"')
    return chat


# In[31]:


def getFeatureVector(chat):
    chat = processTweet(chat)
    featureVector = []
    # split tweet into words
    words = chat.split()
    for w in words:
        featureVector.append(w.lower())
    return " ".join(list(featureVector))


# woRD eMBEDDING USING gLOVE

# In[32]:


def emb_mat(nb_words):
    # may need to download the referenced file below
    EMBEDDING_FILE = "glove.6B.100d.txt"

    def get_coefs(word, *arr):

        return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.strip().split())
                            for o in open(EMBEDDING_FILE, encoding="utf8"))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    emb_mean, emb_std

    embedding_matrix = np.random.normal(
        emb_mean, emb_std, (nb_words+1, embed_size))
    for word, i in word_index.items():
        if (i >= max_features) or i == nb_words:
            continue
        # here we will get embedding for each word from GloVe
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


# tokenization
#

# In[33]:


def tokenized_data(questions, answers, VOCAB_SIZE, tokenizer):
    # encoder_input_data
    import numpy as np
    tokenized_questions = tokenizer.texts_to_sequences(questions)
    maxlen_questions = max([len(x) for x in tokenized_questions])
    padded_questions = preprocessing.sequence.pad_sequences(
        tokenized_questions, maxlen=maxlen, padding='post')
    encoder_input_data = np.array(padded_questions)
    #print( encoder_input_data.shape , maxlen_questions )

    # decoder_input_data
    tokenized_answers = tokenizer.texts_to_sequences(answers)
    maxlen_answers = max([len(x) for x in tokenized_answers])
    padded_answers = preprocessing.sequence.pad_sequences(
        tokenized_answers, maxlen=maxlen, padding='post')
    decoder_input_data = np.array(padded_answers)
    #print( decoder_input_data.shape , maxlen_answers )

    # decoder_output_data
    tokenized_answers = tokenizer.texts_to_sequences(answers)
    for i in range(len(tokenized_answers)):
        # remove <start> take rest
        tokenized_answers[i] = tokenized_answers[i][1:]
    padded_answers = preprocessing.sequence.pad_sequences(
        tokenized_answers, maxlen=maxlen, padding='post')
    onehot_answers = utils.to_categorical(padded_answers, VOCAB_SIZE)
    decoder_output_data = np.array(onehot_answers)
    #print( decoder_output_data.shape )

    return [encoder_input_data, decoder_input_data, decoder_output_data, maxlen_answers]


# DATA PREPARATION

# In[34]:


# define a savepoint for running the model
filepath = "model_Translate_new1.h5"
checkpoint = ModelCheckpoint(
    filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


def prepare_data(questions, answers):
    answers = pd.DataFrame(answers, columns=["Ans"])
    questions = pd.DataFrame(questions, columns=["Question"])
    questions["TokQues"] = questions["Question"].apply(getFeatureVector)

    answers = np.array(answers["Ans"])
    questions = np.array(questions["TokQues"])

    answers_with_tags = list()
    for i in range(len(answers)):
        if type(answers[i]) == str:
            answers_with_tags.append(answers[i])
        else:
            print(questions[i])
            print(answers[i])
            print(type(answers[i]))
            questions.pop(i)

    answers = list()
    for i in range(len(answers_with_tags)):
        answers.append('<START> ' + answers_with_tags[i] + ' <END>')

    tokenizer = preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(questions+answers)

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))

    # embedding_matrix=emb_mat(nb_words)[0]
    # emb_vec=emb_mat(nb_words)[1]

    VOCAB_SIZE = len(tokenizer.word_index)+1

    tok_out = tokenized_data(questions, answers, VOCAB_SIZE, tokenizer)
    encoder_input_data = tok_out[0]
    decoder_input_data = tok_out[1]
    decoder_output_data = tok_out[2]
    maxlen_answers = tok_out[3]

    return [encoder_input_data, decoder_input_data, decoder_output_data, maxlen_answers, nb_words, word_index, tokenizer]


# TRAINING THE DATA
#

# In[35]:


Prepared_data = prepare_data(questions_for_token, answers_for_token)
encoder_input_data = Prepared_data[0]
decoder_input_data = Prepared_data[1]
decoder_output_data = Prepared_data[2]
maxlen_answers = Prepared_data[3]
nb_words = Prepared_data[4]
word_index = Prepared_data[5]
tokenizer = Prepared_data[6]
embedding_matrix = emb_mat(nb_words)
encoder_inputs = tf.keras.layers.Input(shape=(None, ))
encoder_embedding = tf.keras.layers.Embedding(
    nb_words+1, embed_size, mask_zero=True, weights=[embedding_matrix])(encoder_inputs)
encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(
    200, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.layers.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(
    nb_words+1, embed_size, mask_zero=True, weights=[embedding_matrix])(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(
    200, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(
    decoder_embedding, initial_state=encoder_states)

decoder_dense = tf.keras.layers.Dense(
    nb_words+1, activation=tf.keras.activations.softmax)
output = decoder_dense(decoder_outputs)

#model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )


#model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit([encoder_input_data , decoder_input_data], decoder_output_data, batch_size=8, epochs=50, callbacks=callbacks_list)


#

# Making Inference

# In[36]:


def make_inference_models():

    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = tf.keras.layers.Input(shape=(200,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(200,))

    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model


def str_to_tokens(sentence: str):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append(tokenizer.word_index[word])
    return preprocessing.sequence.pad_sequences([tokens_list], maxlen=maxlen, padding='post')


# In[38]:
enc_model, dec_model = make_inference_models()


# In[43]:


def translate(query):
    # for _ in range(10):
    try:
        states_values = enc_model.predict(
            str_to_tokens(query))
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = tokenizer.word_index['start']
        stop_condition = False
        decoded_translation = ''
        while not stop_condition:
            dec_outputs, h, c = dec_model.predict(
                [empty_target_seq] + states_values)
            sampled_word_index = np.argmax(dec_outputs[0, -1, :])
            sampled_word = None
            for word, index in tokenizer.word_index.items():
                if sampled_word_index == index:
                    decoded_translation += ' {}'.format(word)
                    sampled_word = word

            if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
                stop_condition = True

            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = sampled_word_index
            states_values = [h, c]

        return(" ".join(decoded_translation.strip().split(" ")[:-1]))
    except:
        return("(I don't understand).Try sumn else.")


#
