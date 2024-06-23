# Identifying-Toxic-Comments

## 1. Author Introduction
### 1.1. Griffin
<p>Griffin (kleveri2@msu.edu) is a 1st year PHD student at Michigan State University. Griffin’s primary
field of study is within wireless communications,
but he also branches out to many other fields, such
as NLP. Griffin is a teaching assistant in MSU’s capstone class, where he helps seniors along their final
projects before they receive their degrees. When
Griffin is not at school, he enjoys running around
in many of East Lansing’s parks. Griffin will be
focusing his efforts on using the dataset and the
embedding module on this project.</p>

### 1.2. Jotham
<p>Jotham (teshomej@msu.edu) is a 1st year master’s
student at Michigan State University. Jotham is
new to the machine learning space, but has already
found a great deal of interest within the field of
NLP. For this project, Jotham will be working on
the neural network. When Jotham is not doing
classwork, he can usually be found working on one
of the various side projects he’s started.</p>

## 2. Introduction
<p>Online toxicity is a growing issue in the modern day. Ever since the creation of the first chat
rooms, people have used online anonymity to speak
cruel things to one another. This is largely due to
the Online Disinhibition Effect, which states that,
while online, humans have lower behavioral inhibitions. Meaning, they are more likely to portray
aggressive behaviors, such as flaming or inciting
violence when online (Lapidot-Lefler and Barak,
2012). Negative comments on the internet cannot
only lead to hurt feelings or harmed mental health,
but they can easily expand into larger, physical
threats. Due to this, it is more important than ever
that online moderators detect and remove toxicity
as it happens.</p>

<p>Language is constantly morphing. Words can be
forgotten, new words can be made, and old words
can be given different meanings. Due to this, it
is not enough to use a simple toxic word bank to
find toxic words. Indeed, online moderators must
be able to adapt to the introduction of new, never
before seen words and understand their toxicity
from context alone. Due to this, it is essential that
powerful AI based toxicity detection methods exist
to help repel aggressive online behaviors.
</p>

<p>In this project, we set out to create and train a
neural network that will not only detect if a message is toxic, but also which word makes it toxic.
With this in place, newly created toxic words or
words that have received new toxic definitions will
be found without needing to be defined first. Finally, we will also morph the statement to be censored. This will create an easy pipeline for online moderators to censor aggressive words without
needing to explicitly flag specific words or phrases
while still allowing messages to flow.</p>

## 3. Related Works
<p>In Risch and Krestel (2020), the authors give an
in-depth review on the etiquette of the internet and
the classes of online toxicity (Risch and Krestel,
2020). The authors state several unwritten rules of
the internet, the most important to toxicity being:
no personal insults, no discrimination or defamation, and no publishing of personal data. They also
describe the 10 classes of toxicity, which are: profanity, insults, threats, hate speech, and otherwise
toxic comments. The authors then give a summary
of the major NLP models used for detecting toxicity, namely: Word2Vec, GloVE, and Fast Text.</p>

<p>In van Aken et al. (2018), the authors analyze the
errors made by different classifiers when classifying text (van Aken et al., 2018). The authors identify different challenges common in Natural Language Processing tasks such as out-of-vocabulary
words, long-range dependencies, and multi-word
phrases that are present in both the Wikipedia Talk
Pages dataset and the Twitter dataset, both of which
they use in their work. They then describe their
methods to use different types of classifiers, which
include linear regression, bidirectional RNN, and
CNN, to detect toxic comments. Lastly, the authors
provide metrics for their findings, describing error classes to address why a comment was falsely
marked as toxic or non-toxic.</p>

<p>In Kurita et al. (2019), the authors investigate
how well state-of-the-art toxic content classifiers
respond to adversarial attacks (Kurita et al., 2019).
The authors claim that if minimal effort attacks
(e.g., "idiot" to "id*ot") are able to fool automated
text classifiers, these classifiers would become
much less useful. To address the vulnerabilities
they find within modern classifiers, they propose
defenses such as adversarial training and the Contextual Denoising Autoencoder. The authors then
provide a summary of the effects of the defenses
when predicting the class of different toxic comments. In their findings, they describe the strengths
and weaknesses of both of the defense mechanisms
against adversarial attacks.</p>

<p>In Chakrabarty (2020), the author discussed a
method of determining which of the possible toxicity categories a comment falls under (Chakrabarty,
2020). To do this, the author utilizes 3 preprocessing methods and 6 pipelines. The author first
removes all punctuation, then performs lemmatisation, then removes all stop words, commonly used
words that won’t add value. The author uses 6
pipelines for data classification, one for each possible classification. Each pipeline shares the first two
steps: first they create a Bag-of-Words using a word
count vectorizer then they use a tf-idf transformer
on that bag of words. The final step of the pipelines
are different: half use a Support Vector Machine
Model (SVM), and the other half use a Decision
Tree Classifier. Each pipeline is then trained differently for its respective label. The author is able to
receive a 98.08 percent mean validation accuracy</p>

<p>In Georgakopoulos et al. (2018) the authors use
a Convolutional Neural Network (CNN) to classify
toxicity in comments (Georgakopoulos et al., 2018).
CNNs are special types of neural networks that specialize in the classification of data. They use special
layers when compared to normal neural networks:
convolutional layers, pooling layers, embedding
layers, and fully-connected layers. CNNs are traditionally used for image classification, so text inputs
must be modified for proper usage. Words must
be mapped with integers from a vocabulary and
converted to raw integers for a CNN to be usable.
After this conversion, they may be fed into the CNN
as normal. Words will be given embedded using
any of the many embedding algorithms (Word2Vec,
FastText, etc.). The authors performed a series of
experiments comparing CNNs to a kNN, LDA, NB,
and an SVM. They found that the CNN had the
best results, with a mean accuracy of 91.2 percent,
proving that CNNs are an interesting direction for
the future.</p>

<p>In Naim et al. (2022) the authors propose a
framework to capture a toxic fragment within a
toxic span (Naim et al., 2022). The motiviation behind this proposed framework is that current stateof-the-art approaches will classify an entire text as
toxic, while only a small portion of the content may
actually be offensive. To implement their framework, the authors use the Spark NLP Named Entity
Recognition (NER) model, spaCy NER model with
custom toxic tags, and ALBERT NER model to detect the toxic spans within the content. First, the
authors manipulate the input text into a form that
the three NER models can accept, with the spaCy
model accepting input text in the spaCy entity format and both the Spark NLP and ALBERT NER
models accepting BIO (Beginning-Inside-Outside)
annotated tokens. The authors then feed the input
data into the three models, and receive the toxic
spans and the positions of the toxic entities from
the models. From here, the authors use a fusion
technique on the received spans, before appending
space characters between toxic entities within the
span to retrieve the final toxic span from the input
text. The dataset the authors used in their experiments came from the SemEval-2021 toxic spans
dataset, which originate from the Civil Comments
dataset.</p>

<p>In Karimi et al. (2021) the authors use both a
CharacterBERT and Bag-of-Words Model to detect toxic spans(Karimi et al., 2021). The authors
approach the problem as a sequence labelling task,
where they assign one of three predefined classes
to each word in the input row. These labels are B,
I, O which represent the beginning word of a toxic
span, a word within a continuous toxic span, or a
word that is not toxic respectively. These labeled
words are then fed into the models. The use of
CharacterBERT in this case was to make it more
suitable for scenarios where a word is misspelled,
as many toxic comments on the internet tend to
do. As for the Bag-of-Words approach, the authors
claim that its performance in detecting toxic spans
is near the performance of CharacterBERT. The
Bag-of-Words approach involves first building a
dictionary of toxic words based on their frequency
in the training set. Then, going through the test set,
the toxic words are located within each sentence.
After this, the authors use a toxicity ratio, which is
calculated by dividing the frequency of a specific
word by the total frequency of toxic words, to determine whether a word should be labeled as toxic.
An edge case that the authors also covered when
using the Bag-of-Words approach is with ’bleeped’
words. When encountering these bleeped words,
the authors directly assign them as toxic, as there
would be no reason for them to be bleeped otherwise. After testing different model configurations,
including CharacterBERT, BoW, and multiple variations of CharacterBERT + BoW, the authors found
that their best performing model was a combination of the CharacterBERT and BoW approaches,
specifically a variation which they called CharacterBERT + BoW (v1) where the parameters of the
BoW model were set to a minimum term frequency
of 40 and a minimum toxicity ratio of 0.7.</p>

## 4. Methodologies
### 4.1. Dataset
<p>To train our toxicity model, we use the Civil Comments dataset. This dataset is comprised of many
different comments taken from various locations
on the internet. Each comment is scored between 0
and 1 in several categories. These categories are: if
a comment was severely toxic, an insult, obscene,
a threat, an identity attack, or sexually explicit
(Borkan et al., 2019). The different scores provide
an interesting challenge. If scores are viewed as a
percentage, it is difficult to know when a sentence
is truly toxic. Does a sentence with a .10 in toxicity
count as a toxic comment? Does a comment that
has a .20 in insult? Some may say anything above
a .10 is toxic while other may say anything below
a .50 is completely fine for internet usage. This
gray zone makes it difficult to confidently train an
algorithm to detect toxicity. For this paper, any
comment with a .50 or larger in any one category
is considered toxic to insure that only truly toxic
statements are observed. The vast majority of the
comments are clean comments (nearly 95 percent
of comments are clean), so we must be very careful
to avoid over training on the clean comments. This
creates an interesting challenge for classification.</p>

<p>We also determine what toxicity class a sentence
belongs to. The labels provided by the dataset that
we use for our multi-class sentence classification
are ’obscene’, ’insult’, ’threat’, ’identity attack’,
’sexually explicit’, and ’severe toxicity’. Following
a similar idea to assigning binary labels to each
sentence, we only assign a label to a sentence if it
has a toxicity of .50 for that label. To be sure we
are assigning the right label to each class, we only
assign the label with the highest toxicity level that
is over .50. This means if a sentence is labeled as a
threat with a 0.67 toxicity as well as being labeled
as an insult with a 0.55 toxicity, we will label the
sentence as a threat. This method allows us to only
give the best class label to a sentence, which will
allow us to better classify new sentences down the
line.</p>

<p>Apart from our sentence classification, we also
attempt to classify words as either obscene or clear.
To do so, every time we encounter a sentence that
we have counted as obscene, we mark all of the
words that appear in the sentence as ’possibly obscene’. We only label them as possibly obscene
because the words themselves may not necessarily
be obscene in nature, however, we know that there
must be a word within the sentence that is obscene.
Given the number of times a word appears in a
clear sentence vs. an obscene one, we can then
mark the word as belonging to the obscene class or
the clear class.</p>

### 4.2. FastText
<p>We use FastText to obtain embeddings for each
word in our dataset. FastText has the unique ability to retrieve sub-embeddings from words . This
insures that users cannot use slight variation of
words to mask their toxicity (Risch and Krestel,
2020). For example, the word "where" would be
broken into various sub words, like "wh", "whe",
"her", "ere", and "re" (Bojanowski et al., 2017).
From here, each of these sub-embeddings would
be used for the embedding of the overall word.
This means that a toxic word like "loser" would be
split up into sub-embeddings like "lo", "os", "er",
"ser", "lose". If an online agressor were to get
around using a "toxic word" by saying "l0ser", the
sub-embeddings would instead be "l0", "0s", "er",
"ser", "l0se". Note that the sub headings between
the 2 are quite close, and would result in similar
embeddings, and thus be seen as similar. Thus,
FastText is uniquely qualified for picking up new,
unknown, or modified terms and the best suited
for finding toxicity in online comments (Risch and
Krestel, 2020).
</p>

### 4.3. Workflow
<p>The Civil Comments dataset has just over one million total entries with a total sum of around 98
million words within them. The first step in our
workflow is analyzing each of these words to get
the embeddings. For this, we use nltk’s word tokenize() function to split each data point into its
tokens, then we join the tokens back together with a
whitespace separator. After this, we feed all of the
data points into the FastText model. The FastText
model will break each word into subwords and get
their sub-embeddings. These sub-embeddings create a numerical summary of each word. FastText
can obtain any number of n-gram respresentations,
meaning the embeddings take into account n-1 previous words. We observe unigram, bigram, trigram,
and quadgram embeddings, in other words we look
at just the word, the previous word, the previous
two words, or the previous 3 words in different
tests. Then, we simply parse the dataset and feed
our rejoined tokenized comments comments into
the model. We train the FastText model on all
words in the training split of the dataset.</p>

<p>From here, the dataset comments are parsed and
given labels. Any comment with a obscene, insult,
severely toxic, or sexual explicit score above .5
is given the label "obscene". Anything below is
given the label of "clear". For each comment, each
word within that comment is also given a label,
either "obscene" or "clear" based on the label the
comment it originated from. We use our rejoined
tokenized comments method from the FastText embedding section here as well. Only new lines and
line breaks are removed. This is because many
toxic words have symbols in them, like "l0ser" or
"he11", so all symbols are required.</p>

#### 4.3.1. Sentence Processing & Feature Extraction
<p>As stated above, there are two steps in processing
our comments, processing sentences, and processing words. To process the sentences, we first take
a tokenized comment and retrieve its sentence embedding from the FastText model. We then parse
through each word within the sentence to search
for important features. The features we look for
include whether the sentence contains an alphanumeric word, a ’!’ symbol, a ’@’ symbol, a ’#’
symbol, a ’$’ symbol, a ’%’ symbol, a ’ˆ’ symbol, a ’&’ symbol, and a ’*’ symbol. Depending
on if these features appear in the comment or not,
we can create a vector of these features. We then
append the feature vector to our existing sentence
embedding. From here, we determine the given
class for our sentence. For our binary classification
sentence model, we just need to know if any of the
labels are above the 0.5 threshold we set. If that
threshold is met, we label the sentence as obscene,
otherwise, it is labeled as clear.</p>

<p>For our multi-class sentence classification model,
we determine the class for a given sentence based
on the label with the maximum toxicity value that
is over 0.5. If none exist, the sentence is labeled as
clear. Here, we also take the first step in classifying
words as being obscene or clear. Regardless of
binary or multi-class sentence classification, if a
sentence has an obscene label, all of the words
within the sentence are added to a list of possibly
obscene words. This is because we know there
must be some word that is making the sentence
toxic, however, we do not yet know which word
is toxic, so we take all of the words in the toxic
sentence. We also keep track of the number of
times a given word has been in a clear or obscene
sentence, which we use later in the word processing
step to determine the class a word belongs to.</p>

#### 4.3.2. Word Processing & Feature Extraction
<p>When we process words, we don’t look at all of the
words we have seen from the dataset, we only look
at the words that have appeared in obscene sentences. This is because we already know that the
words in the clear sentences cannot be contributing
to the toxicity level of the sentence. Going through
the possibly obscene words we collect while processing sentences, we follow a similar steps as we
did to when processing the sentences as we do processing the words. Again, the first step is to retrieve
the embedding from the FastText model. We then
search the word to create a feature vector. The features we search for are the same for the words as
they are for the sentences, that is, we determine if
it is alphanumeric, contains an ’@’, ’#’, ’$’, etc.
From here, we add the feature vector to the word
embedding. We then determine the class the word
belongs to.</p>

<p>As stated above, we collect the frequency at
which each word appears in a clear sentence and an
obscene sentence. Using the frequency of appearance for each word, we assign its label by determining which class it has had more appearances with.
For example, if the word "hello" has appeared in
100 clear sentences and has only appeared in 50
obscene sentences, we would decide that the word
should not be marked as obscene. One thing we
noticed when processing words from our sentences
is that we add each word from an obscene sentence,
which could lead to the same word appearing multiple times. To explore the affects of this, we decided
to have two binary classification word models, one
where we allow a word to appear as many times
as it appears in obscene sentences, and one where
we set a limit to one appearance per word. We perform both tests to ensure we examine the maximum
number of solutions in order to get the best results.</p>

#### 4.3.3. Data Preparation
<p>After we process the sentences and words and assign them their appropriate classes, we move on
to handling the preparation of our data. Because
we have both multi-class and binary classification
models, we take slightly different approaches to
preparing the data for training. We will first discuss our preparation for our binary classification
data. Given the list of obscene and clear data we
collect during our word and sentence processing
steps, we just add an equal amount of entries from
both sets of data, using the minimum size of the two
sets of data. We do this to prevent oversampling,
which could be detrimental to our results. We then
encode our labels from text values to numerical
values, an input that our models can use. Similarly
to with the preparation of our binary classification
data, we have both obscene and clear data for our
multi-class classification. The main difference is
that because there can be much fewer appearances
of certain labels over others, it did not seem practical to take an amount of entries from each of the
sets of data equal to the size of the smallest set
of data. Instead, we first combine all of the obscenely labeled sentences into a single set of data,
then following the same process as with the binary
classification data, take an amount of entries equal
to the smaller of either the obscene or clear data.
We then create one-hot encodings for each of the
sentences given their label. Finally, we shuffle the
data to prevent labels from appearing in a block
one after another</p>

#### 4.3.4. Model Training
<p>Once all of our previous steps are complete, we
can begin training our models. To build our models, we use Keras neural networks. We have four
models that we have created. Three of the models
are built for binary classification, while one is built
for multi-class classification. The architectures of
all of the models are the identical, with an exception for the activation layer and compilation of the
multi-class model. The neural networks are made
with 5 layers. We use a Dense layer with a 128 output units and a ReLU activation function, a Dense
layer with a 64 output units and a ReLU activation
function, a Dense layer with 32 output units and
a ReLU activation function, a Dense layer with
16 output units and a ReLU activation function,
and an activation layer, which differs depending
on whether the model is for binary classification or
multi-class classification.</p>

<p>For our binary classification models, we use a
final Dense layer which has 1 output unit and uses
a sigmoid activation function, and compile them
using binary crossentropy loss and an Adam optimizer with a learning rate of 0.0003. For our multiclass classification model, we use a final Dense
layer with 7 output units and uses a softmax activation function, and compile it using categorical
crossentropy loss and an Adam optimizer with a
learning rate of 0.0003. For our two sentence models, we fit them for 22 epochs using a batch size
of 16. For our two word models, we fit them for
8 epochs using a batch size of 16. For all of the
models, the number of epochs to train for was determined by setting aside data for validation after
each epoch, and determining the point where both
training and validation losses were at a minimum
and training and validation accuracies were at a
maximum.</p>

#### 4.3.5. Model Testing
<p>After our models are trained, we move on to our
tests. For testing, we use the Civil Comments
dataset’s testing split. The comments in this split
have never been seen in our model, not during the
FastText model training or our word and sentence
classifier training. This gives us the best insight
on the performance of our models on brand new
data. Before we can test our models, we first have
to process the comments to be in a format that our
models can use. The steps we take to process our
testing data are identical to those we took when
processing our training data. Once we have the
features and labels for our sentences and our words,
we feed them into our models for evaluation.
</p>

## 5. Evaluation
### 5.1. Overview
<p>For all of our classifiers, unigram, bigram, trigram,
and quadgram tests are performed to give us comparable metrics for which ngram model is best
for determining the toxicity of both sentences and
words.Tables 1-4 illustrate the differences between
the n-gram models in four key categories: Precision, Recall, F1, and Accuracy.
</p>

### 5.2. Binary Sentence Model
<p>The binary sentence model reads in the features
representing an entire sentence and classifies it as
either toxic or non-toxic (clean). There exist interesting trade offs within the evaluation scores of this
model. For example, Trigram has the best precision
at .8219, yet the lowest results in all other metrics.
Regardless, there is a relatively small difference between the 4 metrics, as they only have an accuracy
range of .008. The Unigram model is likely the
best model. While Trigram has the best precision,
Unigram has the best scores for recall (.7815), F1
(.7904), and accuracy (.7928).</p>

### 5.3. Multi-Class Sentence Model
<p>The multi-class sentence model read in the features
representing and entire sentence, and classifies it as
one of several categories: Obscene, Insult, Identity
Attack, Sexually Explicit, or Severely Toxic. The
results of this model are notably lower than the
results of the binary model. This is due to how
similar classes can be. Something that is insulting
will very likely have obscene words. Something
that is severely toxic may very well be sexually
explicit as well. This model predicts the primary
label of a comment, as that may be very important
for future moderators, but the similarity between
tags results in overall lower scores.</p>

<p>When it comes to the best n-gram model, Quadgram has the best precision (.7869) and accuracy
(.7581) and Unigram has the best recall (.7291)
and F1 (.7527), making it difficult to pick a clear
best option. Across the board, scores are extremely
similar, never being more than .01 away in any
metric from any n-gram model. Thus, any n-gram
multi-class sentence model is roughly the same and
equally viable.</p>

### 5.4. Multi-Appearance Word Model
<p>The multi-appearance word model reads in the
features representing one single word within a
sentence and classifies it as either toxic or nontoxic(clean).
</p>

<p>Some online comments have the same word multiple times in one comment. For example, a potential comment could read "I am tired of this stupid
job and my stupid boss", which has two mentions
of the word stupid. This means the dataset would
have two uses of the word stupid. This has the
double edged effect of providing the dataset with
more data but also potentially over sampling the
dataset with these repeated words. Due to this, two
models are created, one for a dataset which allows
copies of a word to exist in a sentence and one with
those extra words removed.</p>

<p>In respect to the n-gram models, Quadgram has
the best precision (.9866), Trigram has the best
recall (.9824), F1 (.9037), and accuracy (.9108),
making it the best overall model for determining
the toxicity of words. The Trigram model is the superior by as much as .02 in some categories, meaning while it is not the best by a wide margin, it is
the best nonetheless.</p>

### 5.5. Single-Appearance Word Model
<p>The single-appearance word model reads in the
features representing one single word within a
sentence and classifies it as either toxic or nontoxic(clean). Unlike the multi-appearance word
model, this model has all duplicate words removed.</p>

<p>The Unigram model has the best precision
(.9332), the Bigram has the best recall (.6257), F1
(.7411), and accuracy (.7814), making it the clear
best option for general use.
</p>

<p>It is worth noting that this model has noticeably worse results than the multi-appearance word
model by as much as .3 in some categories. While
the fear of over sampling for repeated words exists,
it appears to not be an issue. Having the added
words in the dataset far outweights that issue, and
having repeated words is proven to be beneficial
for word identification</p><br/>

<p align="center"><img src="https://github.com/jothamteshome/Identifying-Toxic-Comments/assets/94427010/21333e6b-b851-47ed-932b-ee7ad85d5c8e"/></p>
<p align="center">Table 1: Unigram Evaluation</p>
<br/>

<p align="center"><img src="https://github.com/jothamteshome/Identifying-Toxic-Comments/assets/94427010/1be070f4-a095-4738-91ce-a5b91959f238"/></p>
<p align="center">Table 2: Bigram Evaluation</p>
<br/>

<p align="center"><img src="https://github.com/jothamteshome/Identifying-Toxic-Comments/assets/94427010/6571b410-215a-44c2-952f-9f11fd7440d4"/></p>
<p align="center">Table 3: Trigram Evaluation</p>
<br/>

<p align="center"><img src="https://github.com/jothamteshome/Identifying-Toxic-Comments/assets/94427010/932765e3-6c72-4752-a0d7-e98a5e0182bc"/></p>
<p align="center">Table 4: Quadgram Evaluation</p>
<br/>

### 5.6. Sentence vs. Word Identification
<p>The results show clearly that identifying a word
gives better results than identifying a sentence by
nearly .1-.2 in each metric. This is likely due to
the difference in complexity. The word "dog" is
clearly not toxic, and the word "damn" alone is
very clearly toxic, as nearly everyone will agree it
is a ’swear’ word. Words typically require much
less surrounding information to be considered toxic
or not. Meanwhile, sentences are much more of a
complex structure. The phrase "That damn dog!"
is likely toxic, but the phrase "That damn dog sure
can run!" is now morphed into a potentially nontoxic sentence, as ’damn’ may now be a term of
endearment toward the dog.</p>


### 5.7. Censoring
<p>Finally, we implement a method to censor individual words from toxic sentences. Each word of
every sentence with a toxic label is analyzed and
classified using the toxic word classifier model. If
a word is deemed toxic, it is then replaced with
"****" within that comment. Four *’s are specifically used to keep the removed word as ambiguous
as possible. If it is deemed not toxic, then the word
remains unaffected.</p>

<p>As shown in Table 5, our system can easily identify ’toxic’ words and automatically remove them
without needing a word bank or any other predefined measures besides the models. Many current
online systems require some form of word bank that contains predefined toxic words, but our system is able to learn toxic words from existing online
documents. For example, in row 4 of Table 5, the
word ’pathetic’ is able to be censored despite it
not being a fundamentally toxic word. However,
the system is not perfect, while it can label new
or rarely used words as toxic, it can also falsely
label clean words as toxic. For example, in that
same sentence, ’Feigning’ is censored despite not
being a toxic word. In other censored comments,
words like "he" or "they" are often censored due to
how often they appear within other toxic comments.
While it is always better to be safe than sorry, looking at methods to decrease this false positive rate
will be an interesting direction for the future.</p><br/>

<p align="center"><img src="https://github.com/jothamteshome/Identifying-Toxic-Comments/assets/94427010/f4e59a0c-a313-4385-b729-34cc635dd55b"/></p>
<p align="center">Table 5: Censored Words</p>
<br/>

## 6. Conclusion
<p>As the number of online users grows, so too will
the amount of toxicity online. As toxicity rises, the
need to protect others from harmful comments becomes more and more important. At the scale the
amount of toxicity appears online, it is not practical
for human moderators to monitor every comment
made online. Even with an army of human moderators present, it is not possible for all harmful
comments to be caught and dealt with, as there will
always be something that is missed. The solution
then is to employ an automated system to detect
and censor harmful comments as they are made. In
this paper, we propose an new method to identify
toxic, harmful online comments. Using a pipeline
of tokenized comments fed into a FastText model
to create embeddings, feature extraction methods,
and a multi-layered neural-network, we were able
to develop systems are not only able to accurately
identify toxic comments and determine the primary
type of toxicity present within the comments, but
we also developed a system that can censor toxic
words in comments deemed toxic. Using our automated system brings online moderation of harmful
comments back into the realm of practicality. As it
stands, our system can censor comments automatically, but with some simple modification, it could
also be used to flag comments that do not cross over
a given threshold for automatic censoring, but with
the help of human moderators could be deemed as
toxic.</p>

## 7. Future Directions
<p>As with any neural network, further fine tuning is
always possible. In specific, more advanced architectures such as RNNs, LSTMs, and transformers
could function better than our current neural network structure and give better results in the four
metrics. Different datasets could also be used. The
Civil Comments dataset, while it certainly gave
solid results, could potentially not be the best for
the job. It could be worthwhile to investigate other
datasets and compare their results. In an ideal situation, multiple datasets would be synthesised and
used in order to get the greatest breadth of data
and create the most generalized models. Finally,
methods to lower the false positive rate need to be
scouted. The current model overly censors many
words. Words such as "they" and "him" are censored due to their heavy appearance in toxic documents. Exploring methods to decrease this rate
could be very interesting and useful.</p>

## References
* Piotr Bojanowski, Edouard Grave, Armand Joulin, and
Tomas Mikolov. 2017. Enriching word vectors with
subword information. _Transactions of the association for computational linguistics_, 5:135–146.

* Daniel Borkan, Lucas Dixon, Jeffrey Sorensen, Nithum
Thain, and Lucy Vasserman. 2019. Nuanced metrics
for measuring unintended bias with real data for text
classification. In _Companion proceedings of the 2019
world wide web conference_, pages 491–500.

* Navoneel Chakrabarty. 2020. A machine learning approach to comment toxicity classification. In _Com-
putational Intelligence in Pattern Recognition: Proceedings of CIPR 2019_, pages 183–193. Springer.

* Spiros V Georgakopoulos, Sotiris K Tasoulis, Aristidis G Vrahatis, and Vassilis P Plagianakos. 2018.
Convolutional neural networks for toxic comment
classification. In _Proceedings of the 10th hellenic
conference on artificial intelligence_, pages 1–6.

* Akbar Karimi, Leonardo Rossi, and Andrea Prati. 2021.
Uniparma at semeval-2021 task 5: Toxic spans detection using characterbert and bag-of-words model. In
"_Proceedings of the 15th International Workshop on
Semantic Evaluation (SemEval-2021)_", pages 220–224. "Association for Computational Linguistics".
  
* Keita Kurita, Anna Belova, and Antonios Anastasopoulos. 2019. Towards robust toxic content classification.
_CoRR_.

* Noam Lapidot-Lefler and Azy Barak. 2012. Effects
of anonymity, invisibility, and lack of eye-contact
on toxic online disinhibition. _Computers in human
behavior_, 28(2):434–443.

* Jannatun Naim, Tashin Hossain, Fareen Tanseem,
Abu Nowshed Chy, and Masaki Aono. 2022. Leveraging fusion of sequence tagging models for toxic
spans detection. _Neurocomputing_, 500:688–702.

* Julian Risch and Ralf Krestel. 2020. Toxic comment
detection in online discussions. _Deep learning-based
approaches for sentiment analysis_, pages 85–109.

* Betty van Aken, Julian Risch, Ralf Krestel, and Alexander Löser. 2018. Challenges for toxic comment classification: An in-depth error analysis. _Proceedings
of the 2nd Workshop on Abusive Language Online
(ALW2)_, pages 33–42.
