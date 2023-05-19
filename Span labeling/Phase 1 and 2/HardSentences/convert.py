from sys import stdin , argv
from nltk.tokenize import word_tokenize

dic = {'o': 'O',
        'bn': 'B-Negative',
        'bp': 'B-Positive',
        'in': 'I-Negative',
        'ip': 'I-Positive'}

sentences = []
labels = []
counter = 0

for line in stdin:
    line = line.strip('\n').lower()
    if line:
        li = word_tokenize(line)
        if counter % 2 == 0:
            sentences.append(li)
        else:
            labels.append(li)
    counter += 1
counter__ = 1
for sentence, label_set in zip(sentences, labels):
    counter_ = 1
    print('#' + argv[1] + str(counter__))
    counter__ += 1
    for word, label in zip(sentence, label_set):
        print(str(counter_) + '\t' + word + '\t' + dic[label])
        counter_ += 1
    print('')



