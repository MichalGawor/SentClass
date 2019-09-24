import random

X = open('./utterance_content.txt', 'r')
Y = open('./rawlabel.txt', 'r')
train = open('./train.tsv', 'w')
test = open('./test.tsv', 'w')
o = []
for x, y in zip(X, Y):
    print(x)
    print(y)
    if x[-1] == '\n':
        x = x[:-1]
    if y[-1] == '\n':
        y = y[:-1]
    o.append(x + '\t' + y + '\n')

random.shuffle(o)
length = len(o)
length = 0.8 * length
length = int(length)
o_train = o[:length]
o_test = o[length:]

for record in o_train:
    train.write(record)

for record in o_test:
    test.write(record)
