# the set of reviews for the *same* paper
class ReviewSet:
    def __init__(self, conference=None, forum_id=None, paper_title=None):
        self.conference = conference
        self.forum_id = forum_id
        self.paper_title = paper_title
        self.reviews = []

class Review:
    def __init__(self, review_id=None, review_text_tokenized=None, reviewer=None, rating=None, confidence=None):
        self.review_id = review_id
        self.review_text_tokenized = review_text_tokenized
        self.reviewer = reviewer
        self.rating = rating
        self.confidence = confidence