import datasets
import fasttext
import keras.models
import nltk
import numpy as np
import os
import random
import tensorflow as tf


from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.backend import get_value
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy, Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow.keras import utils


nltk.download('punkt')
nltk.download('stopwords')

# Builds a model with 4 hidden layers,
# plus a dropout layer if it is specified
def buildModel(multiClass=False, activation_neurons=1):
    # Initialize sequential model
    model = Sequential()

    # Add 4 Dense hidden layers
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))

    if multiClass:
        model.add(Dense(activation_neurons, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=.0003),
                      metrics=[CategoricalAccuracy(), Precision(), Recall()])
    else:
        model.add(Dense(activation_neurons, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=.0003),
                      metrics=[BinaryAccuracy(), Precision(), Recall()])

    return model


# Calculate F1 Score and return string containing results
def displayResults(results):
    accuracy = results[1]
    precision = results[2]
    recall = results[3]

    f1_score = (2 * precision * recall) / (precision + recall)

    return f"Precision: {round(precision, 4)}\nRecall: {round(recall, 4)}\nF1: " \
           f"{round(f1_score, 4)}\nAccuracy: {round(accuracy, 4)}\n"


# Processes words found in sentences labelled as obscene and determines
# they should be deemed obscene or not
def processWords(fasttext_model, possibly_obscene_words, word_sentiments):
    # Initialize list of obscene and clear words
    clear_words = []
    obscene_words = []

    # Add an equal amount of words to xwords and ywords to avoid oversampling
    for word in possibly_obscene_words:

        word_embedding = fasttext_model.get_sentence_vector(word)
        word_embedding = word_embedding.tolist()

        # Initialize a dict of word features
        word_features = {'alnum': 0, 'exclaim': 0, 'at_sign': 0, 'hash': 0, 'money': 0,
                         'percent': 0, 'carat': 0, 'and': 0, 'star': 0}

        # Extract features from word
        if word.isalnum():
            word_features['alnum'] = 1
        if '!' in word:
            word_features['exclaim'] = 1
        if '@' in word:
            word_features['at_sign'] = 1
        if '#' in word:
            word_features['hash'] = 1
        if '$' in word:
            word_features['money'] = 1
        if '%' in word:
            word_features['percent'] = 1
        if '^' in word:
            word_features['carat'] = 1
        if '&' in word:
            word_features['and'] = 1
        if '*' in word:
            word_features['star'] = 1

        # Convert sentence feature dict to list
        word_features = [val for val in word_features.values()]

        # Add features to embedding vector
        word_embedding.extend(word_features)

        # If word appears in more clear sentences, label as clear
        if (word_sentiments[word][0]) >= (word_sentiments[word][1]):
            clear_words.append(np.asarray(word_embedding))

        # Otherwise, label as obscene
        else:
            obscene_words.append(np.asarray(word_embedding))

    return clear_words, obscene_words


# Process in sentences from dataset
def processSentences(fasttext_model, dataset_split):
    print("Loading and processing dataset entries...")
    stops = stopwords.words('english')
    full_count = 0
    sum = 0

    word_sentiments = {}

    # Initialize list containing feature vectors of sentences
    obscene_sentences = []
    clear_sentences = []

    # Initialize list of obscene sentences and words in obscene sentences
    possibly_obscene_words = []

    sentence_classifications = {'obscene': [], 'insult': [], 'identity_attack': [],
                                'threat': [], 'sexual_explicit': [], 'severe_toxicity': []}

    # Load Civil Comments dataset
    dataset = datasets.load_dataset("google/civil_comments", split=dataset_split)

    #  This goes through the input file and adds each sentence to either obscene or clear words
    for count, entry in enumerate(dataset):
        full_count = count

        # Convert text to string from dataset, tokenize it, then convert
        # it back to a whitespace-separated string
        entry_text = get_value(entry['text'])
        entry_tokens = nltk.word_tokenize(entry_text)
        cleaned_text = " ".join(entry_tokens)

        # Get the sentence embedding from the fasttext model
        sentence_embedding = fasttext_model.get_sentence_vector(cleaned_text)
        sentence_embedding = sentence_embedding.tolist()

        # Initialize a dict of sentence features
        sentence_features = {'alnum': 0, 'exclaim': 0, 'at_sign': 0, 'hash': 0, 'money': 0,
                    'percent': 0, 'carat': 0, 'and': 0, 'star': 0}


        # Add to sentence features if any of the special characters match
        for word in cleaned_text.split(" "):
            if word.isalnum():
                sentence_features['alnum'] = 1
            if '!' in word:
                sentence_features['exclaim'] = 1
            if '@' in word:
                sentence_features['at_sign'] = 1
            if '#' in word:
                sentence_features['hash'] = 1
            if '$' in word:
                sentence_features['money'] = 1
            if '%' in word:
                sentence_features['percent'] = 1
            if '^' in word:
                sentence_features['carat'] = 1
            if '&' in word:
                sentence_features['and'] = 1
            if '*' in word:
                sentence_features['star'] = 1

        # Convert sentence feature dict to list
        sentence_features = [val for val in sentence_features.values()]

        # Add features to embedding vector
        sentence_embedding.extend(sentence_features)

        # Collect each sub-toxicity level for the sentence
        toxicities = {'obscene': get_value(entry['obscene']), 'insult': get_value(entry['insult']),
                      'identity_attack': get_value(entry['identity_attack']), 'threat': get_value(entry['threat']),
                      'sexual_explicit': get_value(entry['sexual_explicit']),
                      'severe_toxicity': get_value(entry['severe_toxicity'])}
        tox_list = [(label, tox) for label, tox in toxicities.items()]

        # Sort by largest to smallest toxicity level
        tox_list.sort(key=lambda x: x[1], reverse=True)

        # Determine class for multiclass classification
        if tox_list[0][1] >= 0.5:
            sentence_classifications[tox_list[0][0]].append(np.asarray(sentence_embedding))

        # Determine whether sentence deserves to be labelled as obscene
        # for binary classification
        if get_value(entry['obscene']) >= .5 or get_value(entry["insult"]) >= .5 \
                or get_value(entry['sexual_explicit']) >= .5 \
                or get_value(entry['severe_toxicity']) >= .5\
                or get_value(entry['threat']) >= .5\
                or get_value(entry['identity_attack']) >= .5:

            # Add the sentence to the obscene list
            obscene_sentences.append(np.asarray(sentence_embedding))

            # Add each word as an obscene word
            for word in cleaned_text.split(" "):
                if word in stops or word == " ":
                    continue

                if word not in word_sentiments:
                    word_sentiments[word] = [0, 1]
                else:
                    word_sentiments[word][1] += 1

                possibly_obscene_words.append(word)

        # If it should not be labelled as obscene, mark it as clear
        else:

            # Add each clear sentence to the clear list
            clear_sentences.append(np.asarray(sentence_embedding))

            for word in cleaned_text.split(" "):
                if word in stops or word == " ":
                    continue

                if word not in word_sentiments:
                    word_sentiments[word] = [1, 0]
                else:
                    word_sentiments[word][0] += 1

        if count % 100000 == 0 and count != 0:
            print(f'\rRead {count} lines from dataset...', end='', flush=True)

    print(f'\rRead {full_count} lines from dataset...', end='', flush=True)

    return obscene_sentences, clear_sentences, word_sentiments, possibly_obscene_words, sentence_classifications


# Combine portions of obscene and clear data,
# then split into train and test splits
def splitDataset(obscene, clear):
    X = []
    y = []
    for i in range(min(len(obscene), len(clear))):
        X.append(obscene[i])
        y.append("obscene")
        X.append(clear[i])
        y.append("clear")

    encoder = LabelEncoder()
    encoded_y = encoder.fit_transform(y)

    # split into test and train
    return np.asarray(X), np.asarray(encoded_y)


# Split dataset for multiclass classification use
def multiclassDataset(sentence_classifications, clear):
    label_to_id = {label: i for i, label in enumerate(sentence_classifications.keys())}
    label_to_id['clear'] = len(label_to_id)
    obscene = []
    obscene_labels = []
    X = []
    y = []

    # Loop through different label classifications and
    # add them to single matrix X and their label to y
    for label in sentence_classifications:
        obscene.extend(sentence_classifications[label])
        corresponding_labels = [label_to_id[label]] * len(sentence_classifications[label])
        obscene_labels.extend(corresponding_labels)


    for i in range(min(len(obscene), len(clear))):
        X.append(obscene[i])
        y.append(obscene_labels[i])
        X.append(clear[i])
        y.append(label_to_id['clear'])

    # Categorically encode the labels
    encoder = LabelEncoder()
    encoded_y = encoder.fit_transform(y)
    dummy_y = utils.to_categorical(encoded_y, num_classes=len(label_to_id))

    # Shuffle data around to prevent single batch having one class
    zipped = list(zip(X, dummy_y))
    random.shuffle(zipped)
    X, dummy_y = zip(*zipped)

    return np.asarray(X), np.asarray(dummy_y), label_to_id

# Train sentence and word models
def trainModels():
    models_dir = "models"
    fasttext_embedding_dir = "fasttext_embeddings"
    fasttext_model = fasttext.load_model(f"{fasttext_embedding_dir}/tokenized_bigram.bin")

    os.makedirs('models', exist_ok=True)

    # Process data from dataset then create train and test splits
    obscene_sentences, clear_sentences, word_sentiments, possibly_obscene_words, sentence_classifications = \
        processSentences(fasttext_model, 'train')

    # Get dataset for binary classification and multi-class classification
    X_sents_bin, y_sents_bin = splitDataset(obscene_sentences, clear_sentences)
    X_sents_multi, y_sents_multi, label_to_id = multiclassDataset(sentence_classifications, clear_sentences)

    # Binary Sentence Model training
    sentence_model_bin = buildModel()
    sentence_model_bin.fit(X_sents_bin, y_sents_bin, epochs=22, batch_size=16)
    sentence_model_bin.save(f'{models_dir}/Sentence_Model.keras')

    # MultiClass Sentence Model training
    sentence_model_multi = buildModel(multiClass=True, activation_neurons=len(label_to_id))
    sentence_model_multi.fit(X_sents_multi, y_sents_multi, epochs=22, batch_size=16)
    sentence_model_multi.save(f'{models_dir}/Multi_Class_Sentence_Model.keras')

    # Process data for single appearance words and multi-appearance words
    clear_words_multi, obscene_words_multi = processWords(fasttext_model, possibly_obscene_words, word_sentiments)
    clear_words_sing, obscene_words_sing = processWords(fasttext_model, set(possibly_obscene_words), word_sentiments)

    # Create datasets for single appearance and multi-appearance words
    X_words_multi, y_words_multi = splitDataset(obscene_words_multi, clear_words_multi)
    X_words_sing, y_words_sing = splitDataset(obscene_words_sing, clear_words_sing)

    # MultiAppearance Word Model training
    word_model_multi = buildModel()
    word_model_multi.fit(X_words_multi, y_words_multi, epochs=8, batch_size=16)
    word_model_multi.save(f'{models_dir}/Multi_Appearance_Word_Model.keras')

    # SingleAppearance Word Model training
    word_model_sing = buildModel()
    word_model_sing.fit(X_words_sing, y_words_sing, epochs=8, batch_size=16)
    word_model_sing.save(f'{models_dir}/Single_Appearance_Word_Model.keras')
    

def printCensoredSentences(word_model_multi, fasttext_model, num_sentences=10):
    dataset = datasets.load_dataset("google/civil_comments", split='test')

    toxic_count = 0

    for entry in dataset:
        if get_value(entry['obscene']) >= .5 or get_value(entry["insult"]) >= .5 \
                or get_value(entry['sexual_explicit']) >= .5 \
                or get_value(entry['severe_toxicity']) >= .5\
                or get_value(entry['threat']) >= .5\
                    or get_value(entry['identity_attack']) >= .5:
            # Convert text to string from dataset, tokenize it, then convert
            # it back to a whitespace-separated string
            entry_text = get_value(entry['text'])
            entry_tokens = nltk.word_tokenize(entry_text)
            cleaned_text = " ".join(entry_tokens)
            
            scrubbed_text = cleaned_text[:]
            replaced_words = []
            for word in cleaned_text.split(" "):
        
                word_embedding = fasttext_model.get_sentence_vector(word)
                word_embedding = word_embedding.tolist()
        
                # Initialize a dict of word features
                word_features = {'alnum': 0, 'exclaim': 0, 'at_sign': 0, 'hash': 0, 'money': 0,
                                 'percent': 0, 'carat': 0, 'and': 0, 'star': 0}
        
                # Extract features from word
                if word.isalnum():
                    word_features['alnum'] = 1
                if '!' in word:
                    word_features['exclaim'] = 1
                if '@' in word:
                    word_features['at_sign'] = 1
                if '#' in word:
                    word_features['hash'] = 1
                if '$' in word:
                    word_features['money'] = 1
                if '%' in word:
                    word_features['percent'] = 1
                if '^' in word:
                    word_features['carat'] = 1
                if '&' in word:
                    word_features['and'] = 1
                if '*' in word:
                    word_features['star'] = 1
        
                # Convert sentence feature dict to list
                word_features = [val for val in word_features.values()]
        
                # Add features to embedding vector
                word_embedding.extend(word_features)
                
                result = word_model_multi.predict(np.asarray(word_embedding).reshape(1,-1), verbose = 0)
                if result[0][0] >=.5:
                    replaced_words.append(word)
                    word = " " + word + " "
                    scrubbed_text = scrubbed_text.replace(word, " **** ")

            print(scrubbed_text, replaced_words, end="\n\n")
            toxic_count += 1

            if toxic_count > num_sentences:
                return

# Test models on new, never seen data
def testModels():
    models_dir = "models"
    # Load fasttext model
    fasttext_embedding_dir = "fasttext_embeddings"
    fasttext_model = fasttext.load_model(f"{fasttext_embedding_dir}/tokenized_bigram.bin")

    # Load toxic sentence and word models
    sentence_model_bin = keras.models.load_model(f'{models_dir}/Sentence_Model.keras', compile=False)
    sentence_model_multi = keras.models.load_model(f'{models_dir}/Multi_Class_Sentence_Model.keras', compile=False)
    word_model_multi = keras.models.load_model(f'{models_dir}/Multi_Appearance_Word_Model.keras', compile=False)
    word_model_sing = keras.models.load_model(f'{models_dir}/Single_Appearance_Word_Model.keras',compile=False)

    sentence_model_bin.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=.0003),
                  metrics=[BinaryAccuracy(), Precision(), Recall()])
    sentence_model_multi.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=.0003),
                  metrics=[CategoricalAccuracy(), Precision(), Recall()])
    word_model_multi.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=.0003),
                  metrics=[CategoricalAccuracy(), Precision(), Recall()])
    word_model_sing.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=.0003),
              metrics=[BinaryAccuracy(), Precision(), Recall()])

    # Gather test data
    obscene_sentences, clear_sentences, word_sentiments, possibly_obscene_words, sentence_classifications = \
        processSentences(fasttext_model, 'test')

    # Get dataset for binary classification and multi-class classification
    X_sents_bin, y_sents_bin = splitDataset(obscene_sentences, clear_sentences)
    X_sents_multi, y_sents_multi, label_to_id = multiclassDataset(sentence_classifications, clear_sentences)

    # Process data for single appearance words and multi-appearance words
    clear_words_multi, obscene_words_multi = processWords(fasttext_model, possibly_obscene_words, word_sentiments)
    clear_words_sing, obscene_words_sing = processWords(fasttext_model, set(possibly_obscene_words), word_sentiments)

    # Create datasets for single appearance and multi-appearance words
    X_words_multi, y_words_multi = splitDataset(obscene_words_multi, clear_words_multi)
    X_words_sing, y_words_sing = splitDataset(obscene_words_sing, clear_words_sing)

    # Test models
    bin_sentence_eval_metrics = displayResults(sentence_model_bin.evaluate(X_sents_bin, y_sents_bin))
    multi_sentence_eval_metrics = displayResults(sentence_model_multi.evaluate(X_sents_multi, y_sents_multi))
    multi_word_eval_metrics = displayResults(word_model_multi.evaluate(X_words_multi, y_words_multi))
    sing_word_eval_metrics = displayResults(word_model_sing.evaluate(X_words_sing, y_words_sing))

    # Display results
    print(f'\nBinary Sentence Model Metrics\n{"-" * 30}')
    print(bin_sentence_eval_metrics)

    print(f'Multi-Class Sentence Model Metrics\n{"-" * 30}')
    print(multi_sentence_eval_metrics)

    print(f'Multi-Appearance Word Model Metrics\n{"-" * 30}')
    print(multi_word_eval_metrics)

    print(f'Single Appearance Word Model Metrics\n{"-" * 30}')
    print(sing_word_eval_metrics)
    
    
    # Prints out censorted sentences
    #Comment this out to eliminate this function!!
    printCensoredSentences(word_model_multi, fasttext_model, num_sentences=10)


def main():
    # trainModels()
    testModels()

if __name__ == '__main__':
    main()
