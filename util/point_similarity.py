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

def computer_sentence_sim(sentence_list):
    # Encode all sentences
    embeddings = model.encode(sentence_list)

    # Compute cosine similarity between all pairs
    cos_sim = util.pytorch_cos_sim(embeddings, embeddings)

    # Add all pairs to a list with their cosine similarity score
    all_sentence_combinations = []
    for i in range(len(cos_sim) - 1):
        for j in range(i + 1, len(cos_sim)):
            all_sentence_combinations.append([cos_sim[i][j], i, j])

    # Sort list by the highest cosine similarity score
    all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

    print("Top-5 most similar pairs:")
    for score, i, j in all_sentence_combinations:
        print("{:.4f} \n {} \n {}".format(cos_sim[i][j], sentence_list[i], sentence_list[j]))

if __name__ == '__main__':
    reviewset_array = read_input(input_file)
    review_text_tokenized = (reviewset_array[0].reviews)[0].review_text_tokenized
    merged_paragraphs = merge_sentencetokens(review_text_tokenized)
    for paragraph in merged_paragraphs:
        computer_sentence_sim(paragraph)