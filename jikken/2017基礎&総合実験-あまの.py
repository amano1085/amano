
# coding: utf-8

# In[1]:

#（シャープ）以降の文字はプログラムに影響しません。
# 図やグラフを図示するためのライブラリをインポートする。
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
get_ipython().magic('matplotlib inline')
# 数値計算やデータフレーム操作に関するライブラリをインポートする
import numpy as np
import pandas as pd
from pandas.tools import plotting # 高度なプロットを行うツールのインポート
# URL によるリソースへのアクセスを提供するライブラリをインポートする。
# import urllib # Python 2 の場合
import urllib.request # Python 3 の場合
import sklearn #機械学習のライブラリ
from sklearn.decomposition import PCA #主成分分析器


# ## 3種類についてやってみました。  
# 

# ###ーーーーーーーーーーーーーーーーーーーーーーーーーーー 
# ###好きなアイスクリームアンケート    
# ###ーーーーーーーーーーーーーーーーーーーーーーーーーーー

# In[97]:

df = pd.read_csv('icecream_chosa.txt', sep='\s') # データの読み込み
dfs = df.apply(lambda x: (x-x.mean())/x.std(), axis=0).fillna(0)


# In[98]:

from matplotlib.colors import LinearSegmentedColormap
def generate_cmap(colors):
    """自分で定義したカラーマップを返す"""
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)


# In[99]:

gender = [x for x in dfs['gender']]


# In[100]:

#主成分分析の実行
pca = PCA()
pca.fit(dfs.iloc[:, :])
# データを主成分空間に写像 = 次元圧縮
feature = pca.transform(dfs.iloc[:, :])
# 第一主成分と第二主成分でプロットする
plt.figure(figsize=(8, 8))
cm = generate_cmap(['blue', 'red'])
for x, y, name in zip(feature[:, 0], feature[:, 1], dfs.index):
    plt.text(x, y, name, alpha=0.5, size=15)
plt.scatter(feature[:, 0], feature[:, 1], c=gender, alpha=0.8, cmap=cm)
plt.grid()
plt.show()


# In[101]:

# 累積寄与率を図示する
import matplotlib.ticker as ticker
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution ratio")
plt.grid()
plt.show()


# In[102]:

#主成分分析の実行
pca = PCA()
pca.fit(dfs.iloc[:, :].T)
# データを主成分空間に写像 = 次元圧縮
feature = pca.transform(dfs.iloc[:, :].T)
# 第一主成分と第二主成分でプロットする
plt.figure()
for x, y, name in zip(feature[:, 0], feature[:, 1], dfs.columns):
    plt.text(x, y, name, alpha=0.5, size=15)
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8)
plt.grid()
plt.show()


# In[103]:

# 累積寄与率を図示する
import matplotlib.ticker as ticker
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution ratio")
plt.grid()
plt.show()


# In[104]:

# metric は色々あるので、ケースバイケースでどれかひとつ好きなものを選ぶ。
# method も色々あるので、ケースバイケースでどれかひとつ好きなものを選ぶ。
#y_labels.append("1")
from scipy.cluster.hierarchy import linkage, dendrogram
result1 = linkage(dfs.iloc[:, 2:].T, 
                  #metric = 'braycurtis', 
                  #metric = 'canberra', 
                  #metric = 'chebyshev', 
                  #metric = 'cityblock', 
                  metric = 'correlation', 
                  #metric = 'cosine', 
                  #metric = 'euclidean', 
                  #metric = 'hamming', 
                  #metric = 'jaccard', 
                  #method= 'single')
                  method = 'average')
                  #method= 'complete')
                  #method='weighted')
#dendrogram(result1, labels = list(df.iloc[:, 0:1]))
plt.figure(figsize=(8, 8))
dendrogram(result1, orientation='right', labels=list(df.columns), color_threshold=0.8)
plt.title("Dedrogram")
plt.xlabel("Threshold")
plt.grid()
plt.show()


# In[105]:

feature_names = df.columns[1:]
target_names = list(set(df.iloc[1:, ]))
sample_names = df.index
data = df.iloc[:, 1:]
target = df.iloc[:, 1]


# In[106]:

from sklearn import cross_validation as cv
train_data, test_data, train_target, test_target = cv.train_test_split(data, target, test_size=0.5)


# In[107]:

train_data


# In[108]:

# 様々なパラメータ（ハイパーパラメータという）で学習し、分離性能の最も良いモデルを選択する。
parameters = [
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
    {'kernel': ['rbf'],     'C': [1, 10, 100, 1000], 'gamma': [1e-2, 1e-3, 1e-4]},      
    {'kernel': ['poly'],'C': [1, 10, 100, 1000], 'degree': [2, 3, 4, 5]}]


# In[109]:

from sklearn import svm
from sklearn.metrics import accuracy_score
import time
start = time.time()
from sklearn import grid_search

# train_data を使って、SVM による学習を行う
gs = grid_search.GridSearchCV(svm.SVC(), parameters, n_jobs=2).fit(train_data, train_target)

# 分離性能の最も良かったモデルが何だったか出力する
print(gs.best_estimator_)

# モデル構築に使わなかったデータを用いて、予測性能を評価する
pred_target = gs.predict(test_data)
print ("Accuracy_score:{0}".format(accuracy_score(test_target, pred_target)))
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time))


# In[110]:

# 予測結果と、本当の答えを比較する
df = pd.DataFrame(columns=['test', 'pred'])
df['test'] = test_target # 本当の答え
df['pred'] = pred_target # 予測された答え
df.T


# In[111]:

accuracy_scores=[]
for i in range(10):
    train_data, test_data, train_target, test_target = cv.train_test_split(data, target, test_size=0.8)
    start = time.time()
    # train_data を使って、SVM による学習を行う
    gs = grid_search.GridSearchCV(svm.SVC(), parameters, n_jobs=2).fit(train_data, train_target)

    # モデル構築に使わなかったデータを用いて、予測性能を評価する
    pred_target = gs.predict(test_data)
    accuracy_scores.append(accuracy_score(test_target, pred_target))


# In[112]:

print(accuracy_scores)
print(np.average(accuracy_scores))


# ## 考察  
# PCAで第一主成分と第二主成分でプロットしたところ、なんとなく男女(青と赤)が分かれていそうな見た目であった。  
# また、アイスの種類など各要素の相関係数をaverageリンケージで階層的クラスタリングすると、チョコミントとミント、チョコとチョコチップはそれぞれ近い関係にあった。しかしクラスター単位で見ると、それぞれの属する可能性のあるクラスターはあまり近い関係になかった。チョコミントはチョコ関連なのに。  
# つまりチョコミントを食べたいと思う人は、どちらかというとミントが食べたくてチョコミントを頼むのであって、それほど甘いチョコを欲しているわけではなさそうだと推測される。  
# また、2分割のSVM(クロスバリデーションはしてません)で男女の予測を行った結果、およそ半分くらいの精度で男女の予測ができた。
# 精度としては低いので、食べた種類などで男女を予測するのは、今回のデータだけでは難しそうだ。

# In[ ]:




# In[ ]:




# ###ーーーーーーーーーーーーーーーーーーーーーーーーーーー 
# 
# ###ピマ・インディアンの糖尿病診断  
# ###ーーーーーーーーーーーーーーーーーーーーーーーーーーー 

# In[134]:

df = pd.read_csv('pima-indians-diabetes.txt', sep=',') # データの読み込み
dfs = df.apply(lambda x: (x-x.mean())/x.std(), axis=0).fillna(0)


# In[135]:

cls = [x for x in dfs['Class']]


# In[136]:

#主成分分析の実行
pca = PCA()
pca.fit(dfs.iloc[:, :])
# データを主成分空間に写像 = 次元圧縮
feature = pca.transform(dfs.iloc[:, :])
# 第一主成分と第二主成分でプロットする
plt.figure(figsize=(8, 8), dpi=160)
cm = generate_cmap(['blue', 'red'])
for x, y, name in zip(feature[:, 0], feature[:, 1], dfs.index):
    plt.text(x, y, name, alpha=0.5, size=15)
plt.scatter(feature[:, 0], feature[:, 1], c=cls, alpha=0.8, cmap=cm)
plt.grid()
plt.show()


# In[137]:

# 累積寄与率を図示する
import matplotlib.ticker as ticker
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution ratio")
plt.grid()
plt.show()


# In[138]:

#主成分分析の実行
pca = PCA()
pca.fit(dfs.iloc[:, :].T)
# データを主成分空間に写像 = 次元圧縮
feature = pca.transform(dfs.iloc[:, :].T)
# 第一主成分と第二主成分でプロットする
plt.figure()
for x, y, name in zip(feature[:, 0], feature[:, 1], dfs.columns):
    plt.text(x, y, name, alpha=0.5, size=15)
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8)
plt.grid()
plt.show()


# In[139]:

# 累積寄与率を図示する
import matplotlib.ticker as ticker
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution ratio")
plt.grid()
plt.show()


# In[158]:

# metric は色々あるので、ケースバイケースでどれかひとつ好きなものを選ぶ。
# method も色々あるので、ケースバイケースでどれかひとつ好きなものを選ぶ。
#y_labels.append("1")
from scipy.cluster.hierarchy import linkage, dendrogram
result1 = linkage(dfs.iloc[:, :-1].T, 
                  #metric = 'braycurtis', 
                  #metric = 'canberra', 
                  #metric = 'chebyshev', 
                  #metric = 'cityblock', 
                  metric = 'correlation', 
                  #metric = 'cosine', 
                  #metric = 'euclidean', 
                  #metric = 'hamming', 
                  #metric = 'jaccard', 
                  #method= 'single')
                  method = 'average')
                  #method= 'complete')
                  #method='weighted')
#dendrogram(result1, labels = list(df.iloc[:, 0:1]))
plt.figure(figsize=(8, 8))
dendrogram(result1, orientation='right', labels=list(df.columns[:-1]), color_threshold=0.8)
plt.title("Dedrogram")
plt.xlabel("Threshold")
plt.grid()
plt.show()


# In[160]:

feature_names = df.columns[0:]
target_names = list(set(df.iloc[:, ]))
sample_names = df.index
data = df.iloc[:, :]
target = df.iloc[:, -1]


# In[161]:

from sklearn import cross_validation as cv
train_data, test_data, train_target, test_target = cv.train_test_split(data, target, test_size=0.5)


# In[162]:

train_data


# In[163]:

# 様々なパラメータ（ハイパーパラメータという）で学習し、分離性能の最も良いモデルを選択する。
parameters = [
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
    {'kernel': ['rbf'],     'C': [1, 10, 100, 1000], 'gamma': [1e-2, 1e-3, 1e-4]},      
    {'kernel': ['poly'],'C': [1, 10, 100, 1000], 'degree': [2, 3, 4, 5]}]


# In[164]:

from sklearn import svm
from sklearn.metrics import accuracy_score
import time
start = time.time()
from sklearn import grid_search

# train_data を使って、SVM による学習を行う
gs = grid_search.GridSearchCV(svm.SVC(), parameters, n_jobs=2).fit(train_data, train_target)

# 分離性能の最も良かったモデルが何だったか出力する
print(gs.best_estimator_)

# モデル構築に使わなかったデータを用いて、予測性能を評価する
pred_target = gs.predict(test_data)
print ("Accuracy_score:{0}".format(accuracy_score(test_target, pred_target)))
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time))


# In[165]:

# 予測結果と、本当の答えを比較する
df = pd.DataFrame(columns=['test', 'pred'])
df['test'] = test_target # 本当の答え
df['pred'] = pred_target # 予測された答え
df.T


# ###考察
# 申し訳ございません。まともに考える時間が取れなかったのでテキトーになります・・・。
# 今回作成したSVMで脅威の精度100%を叩き出した。しかし主成分分析のプロットを見るとちゃんと分離できるようには見えない。
# 第一主成分、第二主成分における各特徴変量の主成分負荷量などを見てみると原因がわかるかもしれない。  
# また時間があれば(and学生実験で取り扱われる場合)もう少し考察してみます。

# In[ ]:




# In[ ]:




# ###ーーーーーーーーーーーーーーーーーーーーーーーーーーー 
# 
# ###パーキンソン病診断データ  
# ###ーーーーーーーーーーーーーーーーーーーーーーーーーーー 

# In[190]:

df = pd.read_csv('parkinsons.data', sep=',') # データの読み込み


# In[191]:

dfs = df.iloc[: , 1:].apply(lambda x: (x-x.mean())/x.std(), axis=0).fillna(0)


# In[192]:

dfs


# In[193]:

cls = [x for x in dfs['status']]


# In[194]:

#主成分分析の実行
pca = PCA()
pca.fit(dfs.iloc[:, :])
# データを主成分空間に写像 = 次元圧縮
feature = pca.transform(dfs.iloc[:, :])
# 第一主成分と第二主成分でプロットする
plt.figure(figsize=(8, 8), dpi=160)
cm = generate_cmap(['blue', 'red'])
for x, y, name in zip(feature[:, 0], feature[:, 1], dfs.index):
    plt.text(x, y, name, alpha=0.5, size=15)
plt.scatter(feature[:, 0], feature[:, 1], c=cls, alpha=0.8, cmap=cm)
plt.grid()
plt.show()


# In[195]:

# 累積寄与率を図示する
import matplotlib.ticker as ticker
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution ratio")
plt.grid()
plt.show()


# In[196]:

#主成分分析の実行
pca = PCA()
pca.fit(dfs.iloc[:, :].T)
# データを主成分空間に写像 = 次元圧縮
feature = pca.transform(dfs.iloc[:, :].T)
# 第一主成分と第二主成分でプロットする
plt.figure()
for x, y, name in zip(feature[:, 0], feature[:, 1], dfs.columns):
    plt.text(x, y, name, alpha=0.5, size=15)
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8)
plt.grid()
plt.show()


# In[197]:

# 累積寄与率を図示する
import matplotlib.ticker as ticker
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution ratio")
plt.grid()
plt.show()


# In[198]:

feature_names = df.columns[1:]
target_names = list(set(df.iloc[:, 1:]))
sample_names = df.index
data = df.iloc[:, 1:]
target = df.iloc[:, 17]


# In[199]:

from sklearn import cross_validation as cv
train_data, test_data, train_target, test_target = cv.train_test_split(data, target, test_size=0.8)


# In[200]:

train_data


# In[201]:

# 様々なパラメータ（ハイパーパラメータという）で学習し、分離性能の最も良いモデルを選択する。
parameters = [
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
    {'kernel': ['rbf'],     'C': [1, 10, 100, 1000], 'gamma': [1e-2, 1e-3, 1e-4]},      
    {'kernel': ['poly'],'C': [1, 10, 100, 1000], 'degree': [2, 3, 4, 5]}]


# In[202]:

from sklearn import svm
from sklearn.metrics import accuracy_score
import time
start = time.time()
from sklearn import grid_search

# train_data を使って、SVM による学習を行う
gs = grid_search.GridSearchCV(svm.SVC(), parameters, n_jobs=2).fit(train_data, train_target)

# 分離性能の最も良かったモデルが何だったか出力する
print(gs.best_estimator_)

# モデル構築に使わなかったデータを用いて、予測性能を評価する
pred_target = gs.predict(test_data)
print ("Accuracy_score:{0}".format(accuracy_score(test_target, pred_target)))
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time))


# In[203]:

# 予測結果と、本当の答えを比較する
df = pd.DataFrame(columns=['test', 'pred'])
df['test'] = test_target # 本当の答え
df['pred'] = pred_target # 予測された答え
df.T


# ###考察
# 申し訳ありません。こちらも糖尿病のほうと同じ感じです。  
# このデータは病気のプラスマイナスの項目'status'が列の端っこに無いので、どうすればうまく'status'の列を引っ張ってこれるかが難しいと思いました。  
# 多分他のデータはだいたい最初の方か一番後ろにいるので、その辺が頭使うところかなあと思います。  
# ただ1回の実習で果たして配列における要素の抜き出しを全員が理解できるかどうかは、・・・・・・ってところですね。

# In[ ]:




# In[ ]:




# In[ ]:



