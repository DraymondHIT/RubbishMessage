import jieba.posseg as pesg
import pickle

# 加载分类器
f = open('classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()


# 进行特征选择，这里利用分词后的词性作为特征
def get_word_features(sentence):
    data = {}
    seg_list = pesg.cut(sentence)
    for word, tag in seg_list:
        data[tag] = word
    return data


def predict(sentence):
    return classifier.classify(get_word_features(sentence))


print(predict('我是南京人'))