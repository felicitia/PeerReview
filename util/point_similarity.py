import jsonpickle
from sentence_transformers import SentenceTransformer, util

from entity.review_classes import ReviewSet, Review

"""
compute pair-wise sentence similarities in each paragraph
"""

model = SentenceTransformer('paraphrase-distilroberta-base-v1')
import json

input_file = '../dataset/iclr2019_sample.json'


def read_input(input_file):
    with open(input_file) as json_file:
        json_str = json.load(json_file)
        return jsonpickle.decode(json_str)

def merge_sentencetokens(review_text_tokenized):
    merged_paragraphs = []
    for paragraph in review_text_tokenized:
        merged_paragraph = []
        for sentence in paragraph:
            merged_sentence = ''
            for token in sentence:
                merged_sentence += token
                if sentence.index(token) != (len(sentence) - 1):
                    merged_sentence += ' '
            merged_paragraph.append(merged_sentence)
        merged_paragraphs.append(merged_paragraph)
    return merged_paragraphs

def computer_consecutiveSentences_sim(sentence_list):
    # Encode all sentences
    embeddings = model.encode(sentence_list)

    # Compute cosine similarity between all pairs
    cos_sim = util.pytorch_cos_sim(embeddings, embeddings)

    # Add all pairs to a list with their cosine similarity score
    all_consecutive_sentences = []
    for i in range(len(cos_sim) - 1):
        all_consecutive_sentences.append([cos_sim[i][i+1], i, i+1])

    print("similarity of each consecutive pairs:")
    for score, i, j in all_consecutive_sentences:
        if score >= 0.5:
            print("{:.4f} \n {} \n {}".format(cos_sim[i][j], sentence_list[i], sentence_list[j]))

if __name__ == '__main__':
    reviewset_array = read_input(input_file)
    for reviewset in reviewset_array:
        print(reviewset.paper_title)
        for review in reviewset.reviews:
            merged_paragraphs = merge_sentencetokens(review.review_text_tokenized)
            for paragraph in merged_paragraphs:
                computer_consecutiveSentences_sim(paragraph)