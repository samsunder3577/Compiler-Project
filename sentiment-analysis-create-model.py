from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
import os
import re
import pickle

classifier_filename = 'final_classifier.pkl'
vectorizer_filename = 'final_vectorizer.pkl'

reviews_train = []
for line in open('./data/full_train.txt', 'r',encoding="UTF-8"):

    reviews_train.append(line.strip())

reviews_test = []
for line in open('./data/full_test.txt', 'r',encoding="UTF-8"):

    reviews_test.append(line.strip())

# Clean data

REPLACE_NO_SPACE = re.compile(
    r"(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile(r"(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "


def preprocess_reviews(reviews):

    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower())
               for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line)
               for line in reviews]

    return reviews


reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)

# Vectorize reviews

# vectorizer = CountVectorizer(binary=True)
vectorizer = TfidfVectorizer()
vectorizer.fit(reviews_train_clean)
test_set = vectorizer.transform(reviews_train_clean)
validation_set = vectorizer.transform(reviews_test_clean)

# Build classifier

target = [1 if i < 12501 else 0 for i in range(25001)]

# x_train, x_val, y_train, y_val = train_test_split(
#     test_set, target, train_size=0.75
# )

# for c in [0.01, 0.05, 0.25, 0.5, 1]:

#     lr = LogisticRegression(C=c)
#     lr.fit(x_train, y_train)
#     print ("Accuracy for C=%s: %s"
#            % (c, accuracy_score(y_val, lr.predict(x_val))))

# Train model
final_classifier = LogisticRegression(C=1)
final_classifier.fit(test_set, target)
print ("Final Accuracy: %s"
       % accuracy_score(target, final_classifier.predict(validation_set)))

# feature_to_coef = {
#     word: coef for word, coef in zip(
#         cv.get_feature_names(), final_model.coef_[0]
#     )
# }

# for best_positive in sorted(
#         feature_to_coef.items(),
#         key=lambda x: x[1],
#         reverse=True)[:5]:
#     print (best_positive)

# for best_negative in sorted(
#         feature_to_coef.items(),
#         key=lambda x: x[1])[:5]:
#     print (best_negative)

# save the classifier to disk
pickle.dump(final_classifier, open(classifier_filename, 'wb'))

# save the vectorizer to disk
pickle.dump(vectorizer, open(vectorizer_filename, 'wb'))
