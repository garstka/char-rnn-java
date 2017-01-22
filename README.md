# char-rnn-java
Multi-layer vanilla RNN for predicting the next character in a sequence.
Based on:
 - the article by Andrej Karpathy, http://karpathy.github.io/2015/05/21/rnn-effectiveness/
 - min-char-rnn by Andrej Karpathy, https://gist.github.com/karpathy/d4dee566867f8291f086
 - reimplementation and subsequent work by pavelkomarov (same link, comments)

Main functionality:
 - training - takes utf8 text as input, training a model
 - snapshots - saves the net state every now and then
 - sampling - once trained, it can generate similar text by repeatedly predicting the next character