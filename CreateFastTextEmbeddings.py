import fasttext
import nltk
import tensorflow as tf
import datasets

nltk.download('punkt')


# Read in and write dataset entries to text file
def prepareDatasetText():
    dataset = datasets.load_dataset("google/civil_comments", split='train')

    with open('fasttext_data.txt', 'w', encoding='utf-8') as data_file:
        for entry in dataset:
            entry_text = tf.keras.backend.get_value(entry['text'])
            entry_tokens = nltk.word_tokenize(entry_text)
            cleaned_text = " ".join(entry_tokens)

            data_file.write(f"{cleaned_text} ")


# Build FastText model directly from text file if already created,
# or after creating text file
def buildFastTextModel(prepare_dataset=True):
    if prepare_dataset:
        prepareDatasetText()

    fasttext_embedding_dir = "fasttext_embeddings"
    fasttext_model = fasttext.train_unsupervised(f'{fasttext_embedding_dir}/fasttext_data.txt', model='skipgram', wordNgrams=2)

    fasttext_model.save_model(f"{fasttext_embedding_dir}/tokenized_bigram.bin")


def main():
    buildFastTextModel()


if __name__ == '__main__':
    main()