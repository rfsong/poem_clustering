import glob
import json
import numpy as np
import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

PUNCTUATIONS_AND_STOP_WORDS = {
    '。', '，', '、', 'ε', '□', 'ㄋ', '*', '.', '/', '<', '>', '_', '§',
    'ò', 'ā', 'ō', 'α', 'β', 'ε', 'б', 'и', 'н', 'с', 'х', 'ь', '‘', '’',
    '“', '”', '…', '⑵', '⒛', '┭', '┾', '╆', '□', '《', '》', '【', '】',
    'ぇ', 'が', 'け', 'そ', 'ち', 'ぢ', 'づ', 'ど', 'は', 'や', 'ょ', 'シ', 'ソ',
    'ピ', 'ホ', 'ン', 'ヵ', 'ㄇ', 'ㄉ', 'ㄋ', 'ㄏ', 'ㄓ', 'ㄛ', 
    '\ue3ff', '\ue4bf', '！', '（', '）', '０', '１', '２', '８', '：', '；',
    '？', '＿', 'ａ', 'ｈ', 'ｓ', 'ｖ', 'ｗ', '𡼭', '𢍰', 'α', 'н',
    'ａ', 'ｈ', 'ｖ', 'ｗ',  'ａ', 'ｈ', 'ｖ', 'α', 'н', 
}

# Segment and vectorization: poem to bag of words, then encode BOW to vectors.
def _segment(document):
    tokens = []
    for s in document:
        for w in s:
            if w not in PUNCTUATIONS_AND_STOP_WORDS:  
                tokens.append(w)
    return ' '.join(tokens)


# Load the poem corpus
def load_corpus(pattern):
    corpus = []
    for filename in glob.glob(pattern):
        with open(filename) as json_file:
            data = json.load(json_file)
            for p in data:
                try:
                    corpus.append(_segment(p['paragraphs']))
                except KeyError:
                    print("KeyError:")
                    print(p)
    return corpus


# Print top words of each LDA topic.
def print_topic_top_words(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % topic_idx)
        print(' '.join([feature_names[i]
                        for i in topic.argsort()[:-num_top_words - 1:-1]]))


# Styling
def _color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)


def _make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)


# Print top topic of documents:
def lda_output_to_dataframe(lda_output):
    num_docs = lda_output.shape[0]
    num_topics = lda_output.shape[1]
    
    # column names
    topic_names = ["Topic" + str(i) for i in range(num_topics)]

    # index names
    doc_names = ["Doc" + str(i) for i in range(num_docs)]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 5), columns=topic_names, index=doc_names)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic
    return df_document_topic


def print_doc_topic_distribution(df, max_docs):
    return df.head(max_docs).style.applymap(_color_green).applymap(_make_bold)