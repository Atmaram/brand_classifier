# brand_classifier
A supervised classifier to guess brands from item description

With some analysis on the labels and categories, it can be seen
that the maximum number of brands in a category, in the training
set is < 3k. Also, the number of brands, which extend to other
categories is relatively small (about 10%). This allows for
filtering the brands, based on categories, which helps in reducing
the search space.

The roughly 10% odd failed classification due to this optimization,
can be handled latter, once a decent accuracy is reached.

Current model is a Naive Bayes, with no tuning done, and is the
benchmark against which all further classifiers would be tested!


Total dataset: 1 million records
Training dataset: 600k records
Cross Validation dataset: 200k records
Test dataset: 200k records

Stock Naive Bayes performance (Iteration 1):
============================================
~72% accuracy (on cross validation set, not touched the test set yet)

Iteration 2 (Using first token as filter)
=========================================
Correct guesses: 145943
Wrong guesses: 34007
Passed: 20050
Accuracy: 81.1019727702
