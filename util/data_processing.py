import json
import jsonpickle

from entity.review_classes import Review, ReviewSet

"""
read tokenized raw data (e.g., traindev_train_clean.json) and convert to serialized json file in the dataset folder
each json file is an array of entity.ReviewSet class
"""

tokenized_dataset = '../../../iclr-discourse-dataset/review_rebuttal_pair_dataset_debug/traindev_train_clean.json'
output_file = '../dataset/iclr2019_sample.json'
all_reviews = []

def find_reviewset(forum_id):
    for reviewset in all_reviews:
        if reviewset.forum_id == forum_id:
            return reviewset
    return None

def read_tokenized_data():
    with open(tokenized_dataset) as f:
        data = json.load(f)
    all_pairs = data['review_rebuttal_pairs']
    conference = data['conference']
    # print(all_pairs[0].keys()) # output: dict_keys(['index', 'review_sid', 'rebuttal_sid', 'review_text', 'rebuttal_text', 'title', 'review_author', 'forum', 'labels'])
    for current_pair in all_pairs:
        reviewset = find_reviewset(current_pair['forum'])
        if reviewset == None:
            reviewset = ReviewSet(conference, current_pair['forum'], current_pair['title'])
            current_review = Review(current_pair['review_sid'], current_pair['review_text'], current_pair['review_author'],
                                    current_pair['labels']['rating'], current_pair['labels']['confidence'])
            reviewset.reviews.append(current_review)
            all_reviews.append(reviewset)
        else:
            reviewset.reviews.append(Review(current_pair['review_sid'], current_pair['review_text'], current_pair['review_author'],
                                    current_pair['labels']['rating'], current_pair['labels']['confidence']))

def write_all_reviews():
    json_str = jsonpickle.encode(all_reviews)
    with open(output_file, 'w') as outfile:
        json.dump(json_str, outfile)

if __name__ == "__main__":
    read_tokenized_data()
    write_all_reviews()

# TO DO: follow my format from now on (dataset/iclr2019_sample.json) and use sentence-Bert to check sentence similarities.
# need to merge tokenized data to sentences. decided to not save it to file b/c it's redundent and make the file too large.
# so just use tokenized data (since I need tokens for some tasks) and merge to sentence on the fly