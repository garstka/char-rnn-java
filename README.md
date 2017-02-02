# char-rnn-java
Multi-layer vanilla RNN for predicting the next character in a sequence.

## Based on
 - min-char-rnn by Andrej Karpathy, https://gist.github.com/karpathy/d4dee566867f8291f086 and the article
 http://karpathy.github.io/2015/05/21/rnn-effectiveness/
 - reimplementation and subsequent work by pavelkomarov (same link, comments)

## Why
Made as as a term project for an OOP course. The requirements were to use Java, and no external libraries.

## Main functionality
 - training - takes UTF-8 text as input, trains a model
 - snapshots - saves the network state to file every now and then
 - sampling - once trained, it can generate similar text by repeatedly predicting the next character

## Datasets
 - https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare
 - http://cs.stanford.edu/people/karpathy/char-rnn/
 - any sufficiently large file (>1MB)

## Notes
I tested it mainly on the full Shakespeare dataset (4.6MB).
 - A 2 layer network with hidden size around 100 seems to generate the best results.
 - Increasing hidden size any more than that, or adding more layers, generally makes the network fail to adapt very well.
 - Decreasing the sampling temperature to around 0.8 causes the net to make safer predictions, but also generate some overlong sentences.
 - You could probably try to mitigate some of this, but, after all, it is just a vanilla RNN. Using a more sophisticated architecture
 like an LSTM is a must, if you want to generate results that are actually good.
 - Still, I like that for a simple model it does quite well. Sometimes it manages to capture the way of speaking
 of Shakespeare's characters, using very few actual words. Some of the generated names are also pretty elaborate.


    LIMON SLUMENLUN SENCENSO:
    Yee, take mece it a dold's make a slave
    And thing, bear, I may pleast
    Why in dulow witn ming!

## License
Copyright (c) 2017 Matt Garstka, MIT License