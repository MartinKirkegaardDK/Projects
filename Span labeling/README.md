# NLP2022p01g03
ITU Second Year Project 2022 Group 03  
@developed by:
- Martin Kirkegaard (marki@itu.dk)
- Florian Micliuc (flmi@itu.dk)
- Magnus Sverdrup (magsv@itu.dk)
- Markus Kristoffer Sibbesen (mksi@itu.dk)

This project examines the improvement of a neural model when doing span labelling, by using multiple feature extraction techniques. We show that the performance of a neural model for span detection and sentiment polarity classification is improved by the inclusion of hand-engineered linguistic features, such as affixes, part-of-speech tags and lemmatisation groups.

## How to reproduce results 

Since there is some inherent randomness to neural networks, repreducing the exact values will not be possible. To reproduce how we generated them, run the cells in the jupyter notebook: notebook/main.ipynb

To run this, you will need a vector representation file from this website: https://nlp.stanford.edu/projects/glove/

Download the file named glove.6B.zip

Use the 100d file. To be able to properly load it in, you need to add the line: 

400000 100

to the top of the file.

Put the file in the same folder as the notebook.
