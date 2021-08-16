import pandas as pd

data = pd.read_csv(r"..\data\80w.txt", encoding='utf-8', sep='	', header=None)

# 垃圾短信
import jieba

spam = data[data[1] == 1]
spam[2] = spam[2].map(lambda x: ' '.join(jieba.cut(x)))
# spam.head()

# 正常短信
normal = data[data[1] == 0]
normal[2] = normal[2].map(lambda x: ' '.join(jieba.cut(x)))
# normal.head()

spam.to_csv('spam.csv',encoding='utf-8',header=False,index=False,columns=[2])
normal.to_csv('normal.csv',encoding='utf-8',header=False,index=False,columns=[2])