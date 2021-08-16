import nltk.classify.util
from nltk.classify import NaiveBayesClassifier, DecisionTreeClassifier
from nltk.corpus import PlaintextCorpusReader
import random
import jieba
import jieba.posseg as pesg
import pickle

# 构建停用词列表
f = open(r"..\data\stopWord.txt", encoding='utf-8')
line = f.readline().strip()
stop_words = []
while line:
    stop_words.append(line)
    line = f.readline().strip()
f.close()


# 加载短信语料库
message_corpus = PlaintextCorpusReader('./', ['spam.csv', 'normal.csv'])
all_message = message_corpus.words()


# 删除句中的停用词
def delete_stop_word(sentence):
    words = list(jieba.cut(sentence))
    filtered_words = []
    for w in words:
        if w not in stop_words:
            filtered_words.append(w)
    filtered_sentence = ''.join(filtered_words)
    return filtered_sentence


# 进行特征选择，这里利用分词后的词性作为特征
def get_word_features(sentence):
    data = {}
    sentence = delete_stop_word(sentence)
    seg_list = pesg.cut(sentence)
    for word, tag in seg_list:
        if word != ' ':
            data[tag] = word
    return data


# 短信特征进行标记提取
labels_name = ([(message, '垃圾') for message in message_corpus.words('spam.csv')] + [(message, '正常') for message in
                                                                                    message_corpus.words('normal.csv')])
random.seed(7)
random.shuffle(labels_name)

from nltk.classify import accuracy as nltk_accuracy

featuresets = [(get_word_features(sentence), feature) for (sentence, feature) in labels_name]
train_set, test_set = featuresets[2000:], featuresets[:2000]
classifier = NaiveBayesClassifier.train(train_set)

# 保存分类器
f = open('classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()

print('训练准确率：', str(100 * nltk_accuracy(classifier, train_set)) + str('%'))
print('结果准确率：', str(100 * nltk_accuracy(classifier, test_set)) + str('%'))
