# Import Libraries
import numpy as np
import json
import re
import tensorflow as tf
import random
import spacy

nlp = spacy.load('en_core_web_sm')

"""Import JSON file"""

with open('uni_dataset.json') as f:
    intents = json.load(f)

"""
Preprocessing Data Tasks:

- Clean the data
- split them into inputs and target tensors
-build  a tokenizer dictionary and turn sentences into sequences.

The target tensors has a bunch of list with a length of unique title list.

"""


def preprocessing(pre_line):
    pre_line = re.sub(r'[^a-zA-z.?!\']', ' ', pre_line)
    pre_line = re.sub(r'[ ]+', ' ', pre_line)
    return pre_line


# Parsing json data
inputs, targets = [], []
classes = []
intent_doc = {}

for intent in intents['intents']:
    if intent['intent'] not in classes:
        classes.append(intent['intent'])
    if intent['intent'] not in intent_doc:
        intent_doc[intent['intent']] = []

    for text in intent['text']:
        inputs.append(preprocessing(text))
        targets.append(intent['intent'])

    for response in intent['responses']:
        intent_doc[intent['intent']].append(response)


# Tokenize The data
def tokenize_data(input_list):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')

    tokenizer.fit_on_texts(input_list)
    input_sequence = tokenizer.texts_to_sequences(input_list)
    input_sequence = tf.keras.preprocessing.sequence.pad_sequences(input_sequence, padding='pre')

    return tokenizer, input_sequence


# Preprocessing input data
tokenizer, input_tensor = tokenize_data(inputs)


def create_categorical_target(targets):
    word = {}
    categorical_target = []
    counter = 0
    for target in targets:
        if target not in word:
            word[target] = counter
            counter += 1
        categorical_target.append(word[target])

    categorical_tensor = tf.keras.utils.to_categorical(categorical_target, num_classes=len(word), dtype='int32')
    return categorical_tensor, dict((v, k) for k, v in word.items())


# preprocess output data
target_tensor, target_index_word = create_categorical_target(targets)

# print('input shape: {} and output shape: {}'.format(input_tensor.shape, target_tensor.shape))

"""Building the Model"""

# Hyper-parameters
epochs = 50
vocab_size = len(tokenizer.word_index) + 1
embed_dim = 512
units = 128
target_length = target_tensor.shape[1]

# Build RNN Model with tensorflow
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embed_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, dropout=0.2)),
    tf.keras.layers.Dense(units, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(target_length, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(lr=1e-2)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)

# Train the model
model.fit(input_tensor, target_tensor, epochs=epochs, callbacks=[early_stop])


# Predicting The response from trained data
def response(sentence):
    sent_sequence = []
    doc = nlp(repr(sentence))

    # split the input sentences into words
    for token in doc:
        if token.text in tokenizer.word_index:
            sent_sequence.append(tokenizer.word_index[token.text])

        # handle the unknown words error
        else:
            sent_sequence.append(tokenizer.word_index['<unk>'])

    sent_seq = tf.expand_dims(sent_sequence, 0)
    # predict the category of input sentences
    pred = model(sent_seq)

    pred_class = np.argmax(pred.numpy(), axis=1)

    # Choice a random response for predicted sentence
    response, _ = random.choice(intent_doc[target_index_word[pred_class[0]]]), target_index_word[pred_class[0]]
    return response


def chat():
    # print("Start talking with LUBot (type quit to stop)!")

    while True:
        inp = input("\nYou: ")
        if inp.lower() == "quit":
            break

        resp = response(inp)
        # print(resp)

# chat()
