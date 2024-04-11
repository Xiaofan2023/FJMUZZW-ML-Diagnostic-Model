#!/usr/bin/env python
# coding: utf-8

# # 读取数据

# In[1]:


import pandas as pd
filename = 'raw.csv'
raw = pd.read_csv(filename)
raw.drop(columns=['Unnamed: 0'],inplace=True)


# In[2]:


raw


# In[3]:


raw.head()#查看数据前几行，类似于R head函数
raw.info()#类似于str
raw.describe().T#描述性统计并将结果转置


# # EDA

# In[4]:


import matplotlib.pyplot as plt
#直方图
p = raw.hist(bins = 50, figsize = (20,15),color=("darkgrey"))
#plt.savefig('./不同变量的直方图.tiff', dpi=300, bbox_inches='tight')
plt.savefig('不同变量的直方图.pdf')
plt.show()
#密度图
raw.plot(kind='density', subplots=True, layout=(5,4), figsize=(20, 20), sharex=False)  #每行3个图
#不同变量的分布图
#plt.savefig('./不同变量的密度图.tiff', dpi=300, bbox_inches='tight')
plt.savefig('不同变量的密度图.pdf')
plt.show()


# # 相关性分析

# In[5]:


import seaborn as sns
corr = raw.corr()
plt.figure(figsize=(20,20))
#作图，cmap为颜色类型，annot=False不显示数字，vmax最大值，vmin最小值,suqare=true正方形，linewidths划分每个单元格的行的宽度，xticklabels,yticklabels
sns.heatmap(corr,annot=True,vmax=1,vmin=-1,square=True,cmap='winter',annot_kws={'size':15,'weight':'bold', 'color':'black'})
plt.xticks(fontproperties='Times New Roman',fontsize=20,rotation=45) #x轴刻度的字体大小、旋转角度（逆时针）
plt.yticks(fontproperties='Times New Roman',fontsize=20,rotation=45) #y轴刻度的字体大小、旋转角度（逆时针）
plt.savefig('各变量之间的相关性.pdf')
plt.show()
#根据结果，删除强相关性的变量


# # 缺失值分析

# In[6]:


#pip install missingno
import missingno as msno
#msno.matrix(raw, labels=True)   # 矩阵图

msno.bar(raw,fontsize=12, label_rotation=20)                   # 柱状图

#msno.heatmap(raw)               # 热力图
#msno.dendrogram(raw)

#作图

plt.savefig('缺失值分布图.pdf')
#plt.savefig('./result/不同变量的密度图.tiff', dpi=300)
plt.show()


# # 数据清洗

# In[7]:


# 删除不需要的列
#raw.drop(columns=['Overall survival'],inplace=True)

# 删除重复列
raw.drop_duplicates(keep='first',inplace=True)

# 自变量（协变量）
X = raw.drop(columns=['OS_status'])

# 分类变量处理
X_dummy = pd.get_dummies(X,drop_first=True)

# 缺失数据填补
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)# KNN全称: k nearest neighbor（K近邻算法）
imputer.fit(X_dummy)# 拟合
X_imputed = imputer.transform(X_dummy)# 变换
# 将填补后的协变量（数据格式）转为dataframe格式：
X_imputed_frame = pd.DataFrame(columns=X_dummy.columns, data=X_imputed)


# In[8]:


data = X_imputed_frame
display(data)


# # 数据集拆分

# In[9]:


##导入相关函数
#没有下面部分代码在加载statsmodels.api时会报错
import  scipy.signal.signaltools
def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]
scipy.signal.signaltools._centered = _centered

import statsmodels.api as sm
import seaborn as sbn # Seaborn 可实现对统计数据的可视化展示，基于 Python 语言开发，使用 matplotlib 库
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


# In[10]:


##拆分数据集
y = raw["OS_status"]#结局指标
X = data
#数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state= 202311)


# In[11]:


#将总数据集(data)、训练集(X_trained)、测试集(X_tested)均转为DataFrame格式
X_trained = pd.DataFrame(columns=data.columns, data=X_train)

X_tested = pd.DataFrame(columns=data.columns, data=X_test)


# #  对数据集进行标准化处理，将所有变量转为正态分布数据¶

# In[12]:


# 标准化处理
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# # 通过机器学习的方法进行变量的特征性选择

# In[13]:


# 特征选择（嵌入式）
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

#带L2惩罚项的逻辑回归作为基模型的特征选择   
base_model = LogisticRegression(penalty="l2", C=0.01)
selector = SelectFromModel(estimator=base_model)
X_train = selector.fit_transform(X_train,y_train)
X_test = selector.transform(X_test)

X_train.shape,X_test.shape


# # 11种不同机器学习方法

# In[14]:


#没有下面部分代码在加载statsmodels.api时会报错
import  scipy.signal.signaltools
def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]
scipy.signal.signaltools._centered = _centered

import statsmodels.api as sm
import seaborn as sbn

from warnings import filterwarnings
filterwarnings("ignore")


# ##  传统线性回归

# In[15]:


#线性回归
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression

linearReg = LinearRegression().fit(X_train, y_train)# 训练
linearReg.score(X_train, y_train), linearReg.score(X_test, y_test)# 测试


# ##  LASSO回归

# In[16]:


#LASSO

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.001).fit(X_train, y_train)# 训练
lasso.score(X_train, y_train), lasso.score(X_test, y_test)# 测试


# ##  机器学习logistic

# In[17]:


from sklearn.linear_model import LogisticRegression
#拟合模型
loj = LogisticRegression(solver="liblinear")
loj_model = loj.fit(X_train, y_train)
loj_model
loj_model.score(X_train, y_train),loj_model.score(X_test, y_test)


# In[18]:


#训练集交叉验证
scores = cross_val_score(loj_model, X_train, y_train, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[19]:


from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(loj_model, X_train, y_train, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())



# In[20]:


scores['train_prec_macro'].mean(), scores['train_prec_macro'].std()


# In[21]:


scores['train_rec_macro'].mean(),scores['train_rec_macro'].std()


# In[22]:


scores_f1 = cross_val_score(loj_model, X_train, y_train, cv=5, scoring='f1_macro')
scores_f1.mean(), scores_f1.std()


# In[23]:


#测试集交叉验证
scores = cross_val_score(loj_model, X_test, y_test, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[24]:


scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(loj_model, X_test, y_test, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())


# In[25]:


scores['test_prec_macro'].mean(), scores['test_prec_macro'].std()


# In[26]:


scores['test_rec_macro'].mean(),scores['test_rec_macro'].std()


# In[27]:


scores_f1 = cross_val_score(loj_model, X_test, y_test, cv=5, scoring='f1_macro')
scores_f1.mean(), scores_f1.std()


# In[28]:


from sklearn.metrics import plot_confusion_matrix

# disp = plot_confusion_matrix(loj_model, X_test, y_test, values_format='d')
disp = plot_confusion_matrix(loj_model, X_test, y_test, display_labels=['Live','Die'])  #None,'true','pred','all' 

disp.ax_.set_title('confusion matrix of Logistic',fontsize=20)
disp.ax_.set_xlabel('label predicted by  Logistic',fontsize=15)
disp.ax_.set_ylabel('true label',fontsize=15)

# # # 修改图标、字体大小等属性
# # fig,ax = plt.subplots(dpi=60)
# # disp = plot_confusion_matrix(clf, X_test, y_test, normalize=None, display_labels=['benign','malignant'],ax=ax)  #None,'true','pred','all' 
# # disp.ax_.set_title('confusion matrix of LogisticRegression',fontsize=20)
# # disp.ax_.set_xlabel('label predicted by  LogisticRegression',fontsize=20)
# # plt.show()

plt.savefig('./混淆矩阵_logistic.pdf')
plt.show()


# In[29]:


#logistic打印分类报告
y_pred = loj_model.predict(X_test)
print(classification_report(y_test, y_pred))


# In[30]:


# ROC
import matplotlib.pyplot as plt  
# from sklearn import datasets, metrics, model_selection, svm
from sklearn.metrics import plot_roc_curve

f,ax = plt.subplots(dpi=1200)
plot_roc_curve(loj_model , X_test, y_test, ax=ax, name='test roc')
plot_roc_curve(loj_model , X_train, y_train, ax=ax, name='train roc')
plt.plot([0,1],[0,1],":",label='by chance')
plt.legend()
plt.title('ROC on test set of Logistic',fontsize=14)
plt.savefig("./ROC on test set of Logistic.pdf")
plt.show() 


# In[31]:


# k折交叉验证的多条roc曲线画在同一张图上
from sklearn import metrics
from sklearn.metrics import roc_curve,auc

import numpy as np
# 数据
X_ = scaler.transform(X_imputed)#标准化
X_ = selector.transform(X_)#特征筛选

# 交叉验证及画roc图
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(dpi=300)
for i, (train_index, test_index) in enumerate(skf.split(X_, y)):
    loj_model.fit(X_[train_index], y[train_index])# 训练模型
    viz = metrics.plot_roc_curve(loj_model, X_[test_index], y[test_index],
                         name='ROC fold {}'.format(i),
                         alpha=0.5, lw=1, ax=ax)#画roc曲线
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

# # # ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
# # #         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
        lw=2, alpha=0.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="ROC Curve of StratifiedKFold for Logistic")
ax.legend(loc="lower right")
plt.grid()
plt.savefig("./roc of stratifiedKFold for logistic.pdf")
plt.show()


# In[32]:


# 决策曲线分析图（DCA）
import numpy as np
def get_DCA_coords(y_pred,y_true,fn="DCA.png"):
    """
    get coordinates of points of DCA curve
    """
    pt_arr = []
    net_bnf_arr = []
    y_pred = y_pred.ravel()
    for i in range(0,100,1):
        pt = i /100
        #compute TP FP
        y_pred_clip = (y_pred>pt).astype(int)
        TP = np.sum( y_true*np.round(y_pred_clip) )
        FP = np.sum((1 - y_true) * np.round(y_pred_clip))
        net_bnf = ( TP-FP* pt/(1-pt) )/len(y_true)
        pt_arr.append(pt)
        net_bnf_arr.append(net_bnf)
    
    return pt_arr, net_bnf_arr


y_true = y_test

fig = plt.figure(dpi=300)
y_proba = loj_model.predict_proba(X_test)[:,1]
pt_arr, net_bnf_arr = get_DCA_coords(y_proba,y_test)
plt.plot(pt_arr, net_bnf_arr, lw=2, linestyle='-',label='logistic regression')

plt.plot(pt_arr, np.zeros(len(pt_arr)), color='k', lw=1, linestyle='--',label='All negative')

pt_np = np.array(pt_arr)
pr = np.sum(y_true)/len(y_true)# Prevalence
all_pos = pr-(1-pr)*pt_np/(1-pt_np)
plt.plot(pt_arr, all_pos, color='b', lw=1, linestyle='dotted',label='All positive')

plt.xlim([0.0, 1.0])
plt.ylim([-1, 1])
plt.xlabel('Probability Threshold')
plt.ylabel('Net Benefit')
plt.title('DCA of Logistic')
plt.savefig("./DCA of logistic.pdf")
plt.show()


# ##  连续型朴素贝叶斯-GaussianNB 模型

# In[33]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb_model = nb.fit(X_train, y_train)
nb_model
nb_model.score(X_train, y_train),nb_model.score(X_test, y_test)


# In[34]:


#交叉验证
scores = cross_val_score(nb_model, X_train, y_train, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[35]:


from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(nb_model, X_train, y_train, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())


# In[36]:


scores['train_prec_macro'].mean(), scores['train_prec_macro'].std(),


# In[37]:


scores['train_rec_macro'].mean(),scores['train_rec_macro'].std()


# In[38]:


scores_f1 = cross_val_score(nb_model, X_train, y_train, cv=5, scoring='f1_macro')
scores_f1.mean(), scores_f1.std()


# In[39]:


#测试集交叉验证
scores = cross_val_score(nb_model, X_test, y_test, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[40]:


scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(nb_model, X_test, y_test, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())


# In[41]:


scores['test_prec_macro'].mean(), scores['test_prec_macro'].std()


# In[42]:


scores['test_rec_macro'].mean(),scores['test_rec_macro'].std()


# In[43]:


scores_f1 = cross_val_score(nb_model, X_test, y_test, cv=5, scoring='f1_macro')
scores_f1.mean(), scores_f1.std()


# In[44]:


from sklearn.metrics import plot_confusion_matrix

# disp = plot_confusion_matrix(clf, X_test, y_test, values_format='d')
disp = plot_confusion_matrix(nb_model, X_test, y_test, display_labels=['Live','Die'])  #None,'true','pred','all' 

disp.ax_.set_title('confusion matrix of GNB',fontsize=20)
disp.ax_.set_xlabel('label predicted by  GNB',fontsize=15)
disp.ax_.set_ylabel('true label',fontsize=15)

# # # 修改图标、字体大小等属性
# # fig,ax = plt.subplots(dpi=60)
# # disp = plot_confusion_matrix(clf, X_test, y_test, normalize=None, display_labels=['benign','malignant'],ax=ax)  #None,'true','pred','all' 
# # disp.ax_.set_title('confusion matrix of LogisticRegression',fontsize=20)
# # disp.ax_.set_xlabel('label predicted by  LogisticRegression',fontsize=20)
# # plt.show()

plt.savefig('./混淆矩阵_GNB.pdf')
plt.show()


# In[45]:


# 混淆矩阵¶
y_pred = nb_model.predict(X_test)
#GNB打印分类报告
print(classification_report(y_test, y_pred))


# In[46]:


#### ROC
import matplotlib.pyplot as plt  
# from sklearn import datasets, metrics, model_selection, svm
from sklearn.metrics import plot_roc_curve

f,ax = plt.subplots(dpi=1200)
plot_roc_curve(nb_model , X_test, y_test, ax=ax, name='test roc')  ##
plot_roc_curve(nb_model , X_train, y_train, ax=ax, name='train roc') ##
plt.plot([0,1],[0,1],":",label='by chance')
plt.legend()
plt.title('ROC on test set of GNB',fontsize=14)  ##
plt.savefig("./roc on test set of GNB.pdf")
plt.show() 


# In[47]:


# k折交叉验证的多条roc曲线画在同一张图上
from sklearn import metrics
from sklearn.metrics import roc_curve,auc

import numpy as np
# 数据
X_ = scaler.transform(X_imputed)#标准化
X_ = selector.transform(X_)#特征筛选

# 交叉验证及画roc图
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(dpi=300)
for i, (train_index, test_index) in enumerate(skf.split(X_, y)):
    nb_model.fit(X_[train_index], y[train_index])# 训练模型
    viz = metrics.plot_roc_curve(nb_model, X_[test_index], y[test_index],
                         name='ROC fold {}'.format(i),
                         alpha=0.5, lw=1, ax=ax)#画roc曲线
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

# # # ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
# # #         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
        lw=2, alpha=0.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="ROC Curve of StratifiedKFold for GNB")
ax.legend(loc="lower right")
plt.grid()
plt.savefig("./roc of stratifiedKFold for GNB.pdf")
plt.show()


# In[48]:


# 决策曲线分析图（DCA）
import numpy as np
def get_DCA_coords(y_pred,y_true,fn="DCA.png"):
    """
    get coordinates of points of DCA curve
    """
    pt_arr = []
    net_bnf_arr = []
    y_pred = y_pred.ravel()
    for i in range(0,100,1):
        pt = i /100
        #compute TP FP
        y_pred_clip = (y_pred>pt).astype(int)
        TP = np.sum( y_true*np.round(y_pred_clip) )
        FP = np.sum((1 - y_true) * np.round(y_pred_clip))
        net_bnf = ( TP-FP* pt/(1-pt) )/len(y_true)
        pt_arr.append(pt)
        net_bnf_arr.append(net_bnf)
    
    return pt_arr, net_bnf_arr


y_true = y_test

fig = plt.figure(dpi=300)
y_proba = nb_model.predict_proba(X_test)[:,1]
pt_arr, net_bnf_arr = get_DCA_coords(y_proba,y_test)
plt.plot(pt_arr, net_bnf_arr, lw=2, linestyle='-',label='GNB')

plt.plot(pt_arr, np.zeros(len(pt_arr)), color='k', lw=1, linestyle='--',label='All negative')

pt_np = np.array(pt_arr)
pr = np.sum(y_true)/len(y_true)# Prevalence
all_pos = pr-(1-pr)*pt_np/(1-pt_np)
plt.plot(pt_arr, all_pos, color='b', lw=1, linestyle='dotted',label='All positive')

plt.xlim([0.0, 1.0])
plt.ylim([-1, 1])
plt.xlabel('Probability Threshold')
plt.ylabel('Net Benefit')
plt.title('DCA of GNB')
plt.savefig("./DCA of GNB.pdf")
plt.show()


# ##  K邻近算法-KNN

# In[49]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)
knn_model
y_pred = knn_model.predict(X_test)
knn_model.score(X_train, y_train),knn_model.score(X_test, y_test)


# In[50]:


#交叉验证
scores = cross_val_score(knn_model, X_train, y_train, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[51]:


from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(knn_model, X_train, y_train, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())


# In[52]:


scores['train_prec_macro'].mean(), scores['train_prec_macro'].std()


# In[53]:


scores['train_rec_macro'].mean(),scores['train_rec_macro'].std()


# In[54]:


scores_f1 = cross_val_score(knn_model, X_train, y_train, cv=5, scoring='f1_macro')
scores_f1.mean(), scores_f1.std()


# In[55]:


#测试集交叉验证
scores = cross_val_score(knn_model, X_test, y_test, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[56]:


scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(knn_model, X_test, y_test, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())


# In[57]:


scores['test_prec_macro'].mean(), scores['test_prec_macro'].std()


# In[58]:


scores['test_rec_macro'].mean(),scores['test_rec_macro'].std()


# In[59]:


scores_f1 = cross_val_score(knn_model, X_test, y_test, cv=5, scoring='f1_macro')
scores_f1.mean(), scores_f1.std()


# In[60]:


from sklearn.metrics import plot_confusion_matrix

# disp = plot_confusion_matrix(clf, X_test, y_test, values_format='d')
disp = plot_confusion_matrix(knn_model, X_test, y_test, display_labels=['Live','Die'])  #None,'true','pred','all' 

disp.ax_.set_title('confusion matrix of KNN',fontsize=20)
disp.ax_.set_xlabel('label predicted by  KNN',fontsize=15)
disp.ax_.set_ylabel('true label',fontsize=15)

# # # 修改图标、字体大小等属性
# # fig,ax = plt.subplots(dpi=60)
# # disp = plot_confusion_matrix(clf, X_test, y_test, normalize=None, display_labels=['benign','malignant'],ax=ax)  #None,'true','pred','all' 
# # disp.ax_.set_title('confusion matrix of LogisticRegression',fontsize=20)
# # disp.ax_.set_xlabel('label predicted by  LogisticRegression',fontsize=20)
# # plt.show()

plt.savefig('./混淆矩阵_knn.pdf')
plt.show()


# In[61]:


# 混淆矩阵¶
y_pred = knn_model.predict(X_test)
#GNB打印分类报告
print(classification_report(y_test, y_pred))


# In[62]:


# ROC
import matplotlib.pyplot as plt  
# from sklearn import datasets, metrics, model_selection, svm
from sklearn.metrics import plot_roc_curve

f,ax = plt.subplots(dpi=1200)
plot_roc_curve(knn_model , X_test, y_test, ax=ax, name='test roc')  ##
plot_roc_curve(knn_model , X_train, y_train, ax=ax, name='train roc')  ##
plt.plot([0,1],[0,1],":",label='by chance')
plt.legend()
plt.title('ROC on test set of KNN',fontsize=14)  ##
plt.savefig("./roc on test set of KNN.pdf")
plt.show()


# In[63]:


# k折交叉验证的多条roc曲线画在同一张图上
from sklearn import metrics
from sklearn.metrics import roc_curve,auc

import numpy as np
# 数据
X_ = scaler.transform(X_imputed)#标准化
X_ = selector.transform(X_)#特征筛选

# 交叉验证及画roc图
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(dpi=300)
for i, (train_index, test_index) in enumerate(skf.split(X_, y)):
    knn_model.fit(X_[train_index], y[train_index])# 训练模型  ##
    viz = metrics.plot_roc_curve(knn_model, X_[test_index], y[test_index],  ##
                         name='ROC fold {}'.format(i),
                         alpha=0.5, lw=1, ax=ax)#画roc曲线
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

# # # ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
# # #         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
        lw=2, alpha=0.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="ROC Curve of StratifiedKFold for KNN")  ##
ax.legend(loc="lower right")
plt.grid()
plt.savefig("./roc of stratifiedKFold for KNN.pdf")  ##
plt.show()


# In[64]:


# 决策曲线分析图（DCA）
import numpy as np
def get_DCA_coords(y_pred,y_true,fn="DCA.png"):
    """
    get coordinates of points of DCA curve
    """
    pt_arr = []
    net_bnf_arr = []
    y_pred = y_pred.ravel()
    for i in range(0,100,1):
        pt = i /100
        #compute TP FP
        y_pred_clip = (y_pred>pt).astype(int)
        TP = np.sum( y_true*np.round(y_pred_clip) )
        FP = np.sum((1 - y_true) * np.round(y_pred_clip))
        net_bnf = ( TP-FP* pt/(1-pt) )/len(y_true)
        pt_arr.append(pt)
        net_bnf_arr.append(net_bnf)
    
    return pt_arr, net_bnf_arr


y_true = y_test

fig = plt.figure(dpi=300)
y_proba = knn_model.predict_proba(X_test)[:,1]
pt_arr, net_bnf_arr = get_DCA_coords(y_proba,y_test)
plt.plot(pt_arr, net_bnf_arr, lw=2, linestyle='-',label='KNN') ##

plt.plot(pt_arr, np.zeros(len(pt_arr)), color='k', lw=1, linestyle='--',label='All negative')

pt_np = np.array(pt_arr)
pr = np.sum(y_true)/len(y_true)# Prevalence
all_pos = pr-(1-pr)*pt_np/(1-pt_np)
plt.plot(pt_arr, all_pos, color='b', lw=1, linestyle='dotted',label='All positive')

plt.xlim([0.0, 1.0])
plt.ylim([-1, 1])
plt.xlabel('Probability Threshold')
plt.ylabel('Net Benefit')
plt.title('DCA of KNN') ##
plt.savefig("./DCA of KNN.pdf") ##
plt.show()


# ## 支持向量机-SVM

# In[65]:


from sklearn.svm import SVC
svm_model = SVC(kernel = "linear",probability=True).fit(X_train, y_train)
svm_model
svm_model.score(X_train, y_train),svm_model.score(X_test, y_test)


# In[66]:


#交叉验证
scores = cross_val_score(svm_model, X_train, y_train, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[67]:


from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(svm_model, X_train, y_train, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())


# In[68]:


scores['train_prec_macro'].mean(), scores['train_prec_macro'].std()


# In[69]:


scores['train_rec_macro'].mean(),scores['train_rec_macro'].std()


# In[70]:


scores_f1 = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='f1_macro')
scores_f1.mean(), scores_f1.std()


# In[71]:


#测试集交叉验证
scores = cross_val_score(svm_model, X_test, y_test, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[72]:


scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(svm_model, X_test, y_test, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())


# In[73]:


scores['test_prec_macro'].mean(), scores['test_prec_macro'].std()


# In[74]:


scores['test_rec_macro'].mean(),scores['test_rec_macro'].std()


# In[75]:


scores_f1 = cross_val_score(svm_model, X_test, y_test, cv=5, scoring='f1_macro')
scores_f1.mean(), scores_f1.std()


# In[76]:


from sklearn.metrics import plot_confusion_matrix

# disp = plot_confusion_matrix(clf, X_test, y_test, values_format='d')
disp = plot_confusion_matrix(svm_model, X_test, y_test, display_labels=['Live','Die'])  #None,'true','pred','all' 

disp.ax_.set_title('confusion matrix of SVM',fontsize=20)
disp.ax_.set_xlabel('label predicted by  SVM',fontsize=15)
disp.ax_.set_ylabel('true label',fontsize=15)

# # # 修改图标、字体大小等属性
# # fig,ax = plt.subplots(dpi=60)
# # disp = plot_confusion_matrix(clf, X_test, y_test, normalize=None, display_labels=['benign','malignant'],ax=ax)  #None,'true','pred','all' 
# # disp.ax_.set_title('confusion matrix of LogisticRegression',fontsize=20)
# # disp.ax_.set_xlabel('label predicted by  LogisticRegression',fontsize=20)
# # plt.show()

plt.savefig('./混淆矩阵_svm.pdf')
plt.show()


# In[77]:


# 混淆矩阵¶
y_pred = svm_model.predict(X_test)
#GNB打印分类报告
print(classification_report(y_test, y_pred))


# In[78]:


# ROC
import matplotlib.pyplot as plt  
# from sklearn import datasets, metrics, model_selection, svm
from sklearn.metrics import plot_roc_curve

f,ax = plt.subplots(dpi=1200)
plot_roc_curve(svm_model , X_test, y_test, ax=ax, name='test roc')  ##
plot_roc_curve(svm_model , X_train, y_train, ax=ax, name='train roc')  ##
plt.plot([0,1],[0,1],":",label='by chance')
plt.legend()
plt.title('ROC on test set of SVM',fontsize=14)  ##
plt.savefig("./roc on test set of SVM.pdf")
plt.show() 


# In[79]:


from sklearn import metrics
from sklearn.metrics import roc_curve,auc

import numpy as np
# 数据
X_ = scaler.transform(X_imputed)#标准化
X_ = selector.transform(X_)#特征筛选

# 交叉验证及画roc图
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(dpi=300)
for i, (train_index, test_index) in enumerate(skf.split(X_, y)):
    svm_model.fit(X_[train_index], y[train_index])# 训练模型  ##
    viz = metrics.plot_roc_curve(svm_model, X_[test_index], y[test_index],  ##
                         name='ROC fold {}'.format(i),
                         alpha=0.5, lw=1, ax=ax)#画roc曲线
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

# # # ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
# # #         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
        lw=2, alpha=0.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="ROC Curve of StratifiedKFold for SVM")  ##
ax.legend(loc="lower right")
plt.grid()
plt.savefig("./roc of stratifiedKFold for SVM.pdf")  ##
plt.show()


# In[80]:


# 决策曲线分析图（DCA）
import numpy as np
def get_DCA_coords(y_pred,y_true,fn="DCA.png"):
    """
    get coordinates of points of DCA curve
    """
    pt_arr = []
    net_bnf_arr = []
    y_pred = y_pred.ravel()
    for i in range(0,100,1):
        pt = i /100
        #compute TP FP
        y_pred_clip = (y_pred>pt).astype(int)
        TP = np.sum( y_true*np.round(y_pred_clip) )
        FP = np.sum((1 - y_true) * np.round(y_pred_clip))
        net_bnf = ( TP-FP* pt/(1-pt) )/len(y_true)
        pt_arr.append(pt)
        net_bnf_arr.append(net_bnf)
    
    return pt_arr, net_bnf_arr


y_true = y_test

fig = plt.figure(dpi=300)
y_proba = svm_model.predict_proba(X_test)[:,1]
pt_arr, net_bnf_arr = get_DCA_coords(y_proba,y_test)
plt.plot(pt_arr, net_bnf_arr, lw=2, linestyle='-',label='SVM') ##

plt.plot(pt_arr, np.zeros(len(pt_arr)), color='k', lw=1, linestyle='--',label='All negative')

pt_np = np.array(pt_arr)
pr = np.sum(y_true)/len(y_true)# Prevalence
all_pos = pr-(1-pr)*pt_np/(1-pt_np)
plt.plot(pt_arr, all_pos, color='b', lw=1, linestyle='dotted',label='All positive')

plt.xlim([0.0, 1.0])
plt.ylim([-1, 1])
plt.xlabel('Probability Threshold')
plt.ylabel('Net Benefit')
plt.title('DCA of SVM') ##
plt.savefig("./DCA of SVM.pdf") ##
plt.show()


# ## MLP又名多层感知机，也叫人工神经网络（ANN，Artificial Neural Network）

# In[81]:


from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test_scaled[0:5]
mlpc = MLPClassifier().fit(X_train_scaled, y_train)
mlpc.score(X_train, y_train),mlpc.score(X_test, y_test)


# In[82]:


#交叉验证
scores = cross_val_score(mlpc, X_train, y_train, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[83]:


from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(mlpc, X_train, y_train, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())


# In[84]:


scores['train_prec_macro'].mean(), scores['train_prec_macro'].std()


# In[85]:


scores['train_rec_macro'].mean(),scores['train_rec_macro'].std()


# In[86]:


scores_f1 = cross_val_score(mlpc, X_train, y_train, cv=5, scoring='f1_macro')
scores_f1.mean(), scores_f1.std()


# In[87]:


#测试集交叉验证
scores = cross_val_score(mlpc, X_test, y_test, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[88]:


scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(mlpc, X_test, y_test, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())


# In[89]:


scores['test_prec_macro'].mean(), scores['test_prec_macro'].std()


# In[90]:


scores['test_rec_macro'].mean(),scores['test_rec_macro'].std()


# In[91]:


scores_f1 = cross_val_score(mlpc, X_test, y_test, cv=5, scoring='f1_macro')
scores_f1.mean(), scores_f1.std()


# In[92]:


from sklearn.metrics import plot_confusion_matrix

# disp = plot_confusion_matrix(clf, X_test, y_test, values_format='d')
disp = plot_confusion_matrix(mlpc, X_test, y_test, display_labels=['Live','Die'])  #None,'true','pred','all' 

disp.ax_.set_title('confusion matrix of ANN',fontsize=20)
disp.ax_.set_xlabel('label predicted by  ANN',fontsize=15)
disp.ax_.set_ylabel('true label',fontsize=15)

# # # 修改图标、字体大小等属性
# # fig,ax = plt.subplots(dpi=60)
# # disp = plot_confusion_matrix(clf, X_test, y_test, normalize=None, display_labels=['benign','malignant'],ax=ax)  #None,'true','pred','all' 
# # disp.ax_.set_title('confusion matrix of LogisticRegression',fontsize=20)
# # disp.ax_.set_xlabel('label predicted by  LogisticRegression',fontsize=20)
# # plt.show()

plt.savefig('./混淆矩阵_ANN.pdf')
plt.show()


# In[93]:


# 混淆矩阵¶
y_pred = mlpc.predict(X_test)
#GNB打印分类报告
print(classification_report(y_test, y_pred))


# In[94]:


# ROC
import matplotlib.pyplot as plt  
# from sklearn import datasets, metrics, model_selection, svm
from sklearn.metrics import plot_roc_curve

f,ax = plt.subplots(dpi=1200)
plot_roc_curve(mlpc , X_test, y_test, ax=ax, name='test roc')  ##
plot_roc_curve(mlpc , X_train, y_train, ax=ax, name='train roc')  ##
plt.plot([0,1],[0,1],":",label='by chance')
plt.legend()
plt.title('ROC on test set of ANN',fontsize=14)  ##
plt.savefig("./roc on test set of ANN.pdf")
plt.show() 


# In[95]:


# k折交叉验证的多条roc曲线画在同一张图上
from sklearn import metrics
from sklearn.metrics import roc_curve,auc

import numpy as np
# 数据
#X_ = scaler.transform(X_imputed)#标准化
#X_ = selector.transform(X_)#特征筛选

# 交叉验证及画roc图
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(dpi=300)
for i, (train_index, test_index) in enumerate(skf.split(X_, y)):
    mlpc.fit(X_[train_index], y[train_index])# 训练模型  ##
    viz = metrics.plot_roc_curve(mlpc, X_[test_index], y[test_index],  ##
                         name='ROC fold {}'.format(i),
                         alpha=0.5, lw=1, ax=ax)#画roc曲线
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

# # # ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
# # #         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
        lw=2, alpha=0.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="ROC Curve of StratifiedKFold for ANN")  ##
ax.legend(loc="lower right")
plt.grid()
plt.savefig("./roc of stratifiedKFold for ANN.pdf")  ##
plt.show()


# In[96]:


# 决策曲线分析图（DCA）
import numpy as np
def get_DCA_coords(y_pred,y_true,fn="DCA.png"):
    """
    get coordinates of points of DCA curve
    """
    pt_arr = []
    net_bnf_arr = []
    y_pred = y_pred.ravel()
    for i in range(0,100,1):
        pt = i /100
        #compute TP FP
        y_pred_clip = (y_pred>pt).astype(int)
        TP = np.sum( y_true*np.round(y_pred_clip) )
        FP = np.sum((1 - y_true) * np.round(y_pred_clip))
        net_bnf = ( TP-FP* pt/(1-pt) )/len(y_true)
        pt_arr.append(pt)
        net_bnf_arr.append(net_bnf)
    
    return pt_arr, net_bnf_arr


y_true = y_test

fig = plt.figure(dpi=300)
y_proba = mlpc.predict_proba(X_test)[:,1]
pt_arr, net_bnf_arr = get_DCA_coords(y_proba,y_test)
plt.plot(pt_arr, net_bnf_arr, lw=2, linestyle='-',label='MLP') ##

plt.plot(pt_arr, np.zeros(len(pt_arr)), color='k', lw=1, linestyle='--',label='All negative')

pt_np = np.array(pt_arr)
pr = np.sum(y_true)/len(y_true)# Prevalence
all_pos = pr-(1-pr)*pt_np/(1-pt_np)
plt.plot(pt_arr, all_pos, color='b', lw=1, linestyle='dotted',label='All positive')

plt.xlim([0.0, 1.0])
plt.ylim([-1, 1])
plt.xlabel('Probability Threshold')
plt.ylabel('Net Benefit')
plt.title('DCA of ANN') ##
plt.savefig("./DCA of ANN.pdf") ##
plt.show()


# ## 决策树

# In[97]:


from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()
cart_model = cart.fit(X_train, y_train)
cart_model.score(X_train, y_train),cart_model.score(X_test, y_test)


# In[98]:


#交叉验证
scores = cross_val_score(cart_model, X_train, y_train, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[99]:


from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(cart_model, X_train, y_train, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())


# In[100]:


scores['train_prec_macro'].mean(), scores['train_prec_macro'].std()


# In[101]:


scores['train_rec_macro'].mean(),scores['train_rec_macro'].std()


# In[102]:


scores_f1 = cross_val_score(cart_model, X_train, y_train, cv=5, scoring='f1_macro')
scores_f1.mean(), scores_f1.std()


# In[103]:


#测试集交叉验证
scores = cross_val_score(cart_model, X_test, y_test, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[104]:


scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(cart_model, X_test, y_test, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())


# In[105]:


scores['test_prec_macro'].mean(), scores['test_prec_macro'].std()


# In[106]:


scores['test_rec_macro'].mean(),scores['test_rec_macro'].std()


# In[107]:


scores_f1 = cross_val_score(cart_model, X_test, y_test, cv=5, scoring='f1_macro')
scores_f1.mean(), scores_f1.std()


# In[108]:


from sklearn.metrics import plot_confusion_matrix

# disp = plot_confusion_matrix(clf, X_test, y_test, values_format='d')
disp = plot_confusion_matrix(cart_model, X_test, y_test, display_labels=['Live','Die'])  #None,'true','pred','all' 

disp.ax_.set_title('confusion matrix of DecisionTree',fontsize=20)
disp.ax_.set_xlabel('label predicted by  DecisionTree',fontsize=15)
disp.ax_.set_ylabel('true label',fontsize=15)

# # # 修改图标、字体大小等属性
# # fig,ax = plt.subplots(dpi=60)
# # disp = plot_confusion_matrix(clf, X_test, y_test, normalize=None, display_labels=['benign','malignant'],ax=ax)  #None,'true','pred','all' 
# # disp.ax_.set_title('confusion matrix of LogisticRegression',fontsize=20)
# # disp.ax_.set_xlabel('label predicted by  LogisticRegression',fontsize=20)
# # plt.show()

plt.savefig('./混淆矩阵_DecisionTree.pdf')
plt.show()


# In[109]:


# 混淆矩阵¶
y_pred = cart_model.predict(X_test)
#Decision Tree打印分类报告
print(classification_report(y_test, y_pred))


# In[110]:


## ROC
import matplotlib.pyplot as plt  
# from sklearn import datasets, metrics, model_selection, svm
from sklearn.metrics import plot_roc_curve

f,ax = plt.subplots(dpi=1200)
plot_roc_curve(cart_model , X_test, y_test, ax=ax, name='test roc')  ##
plot_roc_curve(cart_model , X_train, y_train, ax=ax, name='train roc')  ##
plt.plot([0,1],[0,1],":",label='by chance')
plt.legend()
plt.title('ROC on test set of Decision Tree',fontsize=14)  ##
plt.savefig("./roc on test set of Decision Tree.pdf")
plt.show() 


# In[111]:


# k折交叉验证的多条roc曲线画在同一张图上
from sklearn import metrics
from sklearn.metrics import roc_curve,auc

import numpy as np
# 数据
#X_ = scaler.transform(X_imputed)#标准化
#X_ = selector.transform(X_)#特征筛选

# 交叉验证及画roc图
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(dpi=300)
for i, (train_index, test_index) in enumerate(skf.split(X_, y)):
    cart_model.fit(X_[train_index], y[train_index])# 训练模型  ##
    viz = metrics.plot_roc_curve(cart_model, X_[test_index], y[test_index],  ##
                         name='ROC fold {}'.format(i),
                         alpha=0.5, lw=1, ax=ax)#画roc曲线
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

# # # ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
# # #         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
        lw=2, alpha=0.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="ROC Curve of StratifiedKFold for Decision Tree")  ##
ax.legend(loc="lower right")
plt.grid()
plt.savefig("./roc of stratifiedKFold for Decision Tree.pdf")  ##
plt.show()


# In[112]:


# 决策曲线分析图（DCA）
import numpy as np
def get_DCA_coords(y_pred,y_true,fn="DCA.png"):
    """
    get coordinates of points of DCA curve
    """
    pt_arr = []
    net_bnf_arr = []
    y_pred = y_pred.ravel()
    for i in range(0,100,1):
        pt = i /100
        #compute TP FP
        y_pred_clip = (y_pred>pt).astype(int)
        TP = np.sum( y_true*np.round(y_pred_clip) )
        FP = np.sum((1 - y_true) * np.round(y_pred_clip))
        net_bnf = ( TP-FP* pt/(1-pt) )/len(y_true)
        pt_arr.append(pt)
        net_bnf_arr.append(net_bnf)
    
    return pt_arr, net_bnf_arr


y_true = y_test

fig = plt.figure(dpi=300)
y_proba = cart_model.predict_proba(X_test)[:,1]
pt_arr, net_bnf_arr = get_DCA_coords(y_proba,y_test)
plt.plot(pt_arr, net_bnf_arr, lw=2, linestyle='-',label='cart') ##

plt.plot(pt_arr, np.zeros(len(pt_arr)), color='k', lw=1, linestyle='--',label='All negative')

pt_np = np.array(pt_arr)
pr = np.sum(y_true)/len(y_true)# Prevalence
all_pos = pr-(1-pr)*pt_np/(1-pt_np)
plt.plot(pt_arr, all_pos, color='b', lw=1, linestyle='dotted',label='All positive')

plt.xlim([0.0, 1.0])
plt.ylim([-1, 1])
plt.xlabel('Probability Threshold')
plt.ylabel('Net Benefit')
plt.title('DCA of Decision Tree') ##
plt.savefig("./DCA of Decision Tree.pdf") ##
plt.show()


# ## 随机森林

# In[113]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
rf_model = RandomForestClassifier().fit(X_train, y_train)
rf_model.score(X_train, y_train),rf_model.score(X_test, y_test)


# In[114]:


#交叉验证
scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[115]:


from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(rf_model, X_train, y_train, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())


# In[116]:


scores['train_prec_macro'].mean(), scores['train_prec_macro'].std()


# In[117]:


scores['train_rec_macro'].mean(),scores['train_rec_macro'].std()


# In[118]:


scores_f1 = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='f1_macro')
scores_f1.mean(), scores_f1.std()


# In[119]:


#测试集交叉验证
scores = cross_val_score(rf_model, X_test, y_test, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[120]:


scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(rf_model, X_test, y_test, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())


# In[121]:


scores['test_prec_macro'].mean(), scores['test_prec_macro'].std()


# In[122]:


scores['test_rec_macro'].mean(),scores['test_rec_macro'].std()


# In[123]:


scores_f1 = cross_val_score(rf_model, X_test, y_test, cv=5, scoring='f1_macro')
scores_f1.mean(), scores_f1.std()


# In[124]:


from sklearn.metrics import plot_confusion_matrix

# disp = plot_confusion_matrix(clf, X_test, y_test, values_format='d')
disp = plot_confusion_matrix(rf_model, X_test, y_test, display_labels=['Live','Die'])  #None,'true','pred','all' 

disp.ax_.set_title('confusion matrix of RandomForest',fontsize=20)
disp.ax_.set_xlabel('label predicted by  RandomForest',fontsize=15)
disp.ax_.set_ylabel('true label',fontsize=15)

# # # 修改图标、字体大小等属性
# # fig,ax = plt.subplots(dpi=60)
# # disp = plot_confusion_matrix(clf, X_test, y_test, normalize=None, display_labels=['benign','malignant'],ax=ax)  #None,'true','pred','all' 
# # disp.ax_.set_title('confusion matrix of LogisticRegression',fontsize=20)
# # disp.ax_.set_xlabel('label predicted by  LogisticRegression',fontsize=20)
# # plt.show()

plt.savefig('./混淆矩阵_RandomForest.pdf')
plt.show()


# In[125]:


y_pred = rf_model.predict(X_test)
#Random Forest打印分类报告
print(classification_report(y_test, y_pred))


# In[126]:


## ROC
import matplotlib.pyplot as plt  
# from sklearn import datasets, metrics, model_selection, svm
from sklearn.metrics import plot_roc_curve

f,ax = plt.subplots(dpi=1200)
plot_roc_curve(rf_model , X_test, y_test, ax=ax, name='test roc')  ##
plot_roc_curve(rf_model , X_train, y_train, ax=ax, name='train roc')  ##
plt.plot([0,1],[0,1],":",label='by chance')
plt.legend()
plt.title('ROC on test set of Random Forest',fontsize=14)  ##
plt.savefig("./roc on test set of Random Forest.pdf")
plt.show() 


# In[127]:


# k折交叉验证的多条roc曲线画在同一张图上
from sklearn import metrics
from sklearn.metrics import roc_curve,auc

import numpy as np
# 数据
#X_ = scaler.transform(X_imputed)#标准化
#X_ = selector.transform(X_)#特征筛选

# 交叉验证及画roc图
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(dpi=300)
for i, (train_index, test_index) in enumerate(skf.split(X_, y)):
    rf_model.fit(X_[train_index], y[train_index])# 训练模型  ##
    viz = metrics.plot_roc_curve(rf_model, X_[test_index], y[test_index],  ##
                         name='ROC fold {}'.format(i),
                         alpha=0.5, lw=1, ax=ax)#画roc曲线
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

# # # ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
# # #         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
        lw=2, alpha=0.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="ROC Curve of StratifiedKFold for Random Forest")  ##
ax.legend(loc="lower right")
plt.grid()
plt.savefig("./roc of stratifiedKFold for Random Forest.pdf")  ##
plt.show()


# In[128]:


# 决策曲线分析图（DCA）
import numpy as np
def get_DCA_coords(y_pred,y_true,fn="DCA.png"):
    """
    get coordinates of points of DCA curve
    """
    pt_arr = []
    net_bnf_arr = []
    y_pred = y_pred.ravel()
    for i in range(0,100,1):
        pt = i /100
        #compute TP FP
        y_pred_clip = (y_pred>pt).astype(int)
        TP = np.sum( y_true*np.round(y_pred_clip) )
        FP = np.sum((1 - y_true) * np.round(y_pred_clip))
        net_bnf = ( TP-FP* pt/(1-pt) )/len(y_true)
        pt_arr.append(pt)
        net_bnf_arr.append(net_bnf)
    
    return pt_arr, net_bnf_arr


y_true = y_test

fig = plt.figure(dpi=300)
y_proba = rf_model.predict_proba(X_test)[:,1]  ##
pt_arr, net_bnf_arr = get_DCA_coords(y_proba,y_test)
plt.plot(pt_arr, net_bnf_arr, lw=2, linestyle='-',label='RF') ##

plt.plot(pt_arr, np.zeros(len(pt_arr)), color='k', lw=1, linestyle='--',label='All negative')

pt_np = np.array(pt_arr)
pr = np.sum(y_true)/len(y_true)# Prevalence
all_pos = pr-(1-pr)*pt_np/(1-pt_np)
plt.plot(pt_arr, all_pos, color='b', lw=1, linestyle='dotted',label='All positive')

plt.xlim([0.0, 1.0])
plt.ylim([-1, 1])
plt.xlabel('Probability Threshold')
plt.ylabel('Net Benefit')
plt.title('DCA of Random Forest') ##
plt.savefig("./DCA of Random Forest.pdf") ##
plt.show()


# ##  GBM模型

# In[129]:


gbm_model = GradientBoostingClassifier().fit(X_train, y_train)
gbm_model.score(X_train, y_train),gbm_model.score(X_test, y_test)


# In[130]:


#交叉验证
scores = cross_val_score(gbm_model, X_train, y_train, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[131]:


from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(gbm_model, X_train, y_train, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())


# In[132]:


scores['train_prec_macro'].mean(), scores['train_prec_macro'].std()


# In[133]:


scores['train_rec_macro'].mean(),scores['train_rec_macro'].std()


# In[134]:


scores_f1 = cross_val_score(gbm_model, X_train, y_train, cv=5, scoring='f1_macro')
scores_f1.mean(), scores_f1.std()


# In[135]:


#测试集交叉验证
scores = cross_val_score(gbm_model, X_test, y_test, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[136]:


scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(gbm_model, X_test, y_test, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())


# In[137]:


scores['test_prec_macro'].mean(), scores['test_prec_macro'].std()


# In[138]:


scores['test_rec_macro'].mean(),scores['test_rec_macro'].std()


# In[139]:


scores_f1 = cross_val_score(gbm_model, X_test, y_test, cv=5, scoring='f1_macro')
scores_f1.mean(), scores_f1.std()


# In[140]:


from sklearn.metrics import plot_confusion_matrix

# disp = plot_confusion_matrix(clf, X_test, y_test, values_format='d')
disp = plot_confusion_matrix(gbm_model, X_test, y_test, display_labels=['Live','Die'])  #None,'true','pred','all' 

disp.ax_.set_title('confusion matrix of GBM',fontsize=20)
disp.ax_.set_xlabel('label predicted by  GBM',fontsize=15)
disp.ax_.set_ylabel('true label',fontsize=15)

# # # 修改图标、字体大小等属性
# # fig,ax = plt.subplots(dpi=60)
# # disp = plot_confusion_matrix(clf, X_test, y_test, normalize=None, display_labels=['benign','malignant'],ax=ax)  #None,'true','pred','all' 
# # disp.ax_.set_title('confusion matrix of LogisticRegression',fontsize=20)
# # disp.ax_.set_xlabel('label predicted by  LogisticRegression',fontsize=20)
# # plt.show()

plt.savefig('./混淆矩阵_GBM.pdf')
plt.show()


# In[141]:


y_pred = gbm_model.predict(X_test)
#GBM打印分类报告
print(classification_report(y_test, y_pred))


# In[142]:


## ROC
import matplotlib.pyplot as plt  
# from sklearn import datasets, metrics, model_selection, svm
from sklearn.metrics import plot_roc_curve

f,ax = plt.subplots(dpi=1200)
plot_roc_curve(gbm_model , X_test, y_test, ax=ax, name='test roc')  ##
plot_roc_curve(gbm_model , X_train, y_train, ax=ax, name='train roc')  ##
plt.plot([0,1],[0,1],":",label='by chance')
plt.legend()
plt.title('ROC on test set of GBM',fontsize=14)  ##
plt.savefig("./roc on test set of GBM.pdf")
plt.show()


# In[143]:


# k折交叉验证的多条roc曲线画在同一张图上
from sklearn import metrics
from sklearn.metrics import roc_curve,auc

import numpy as np
# 数据
#X_ = scaler.transform(X_imputed)#标准化
#X_ = selector.transform(X_)#特征筛选

# 交叉验证及画roc图
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(dpi=300)
for i, (train_index, test_index) in enumerate(skf.split(X_, y)):
    gbm_model.fit(X_[train_index], y[train_index])# 训练模型  ##
    viz = metrics.plot_roc_curve(gbm_model, X_[test_index], y[test_index],  ##
                         name='ROC fold {}'.format(i),
                         alpha=0.5, lw=1, ax=ax)#画roc曲线
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

# # # ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
# # #         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
        lw=2, alpha=0.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="ROC Curve of StratifiedKFold for GBM")  ##
ax.legend(loc="lower right")
plt.grid()
plt.savefig("./roc of stratifiedKFold for GBM.pdf")  ##
plt.show()


# In[144]:


# 决策曲线分析图（DCA）
import numpy as np
def get_DCA_coords(y_pred,y_true,fn="DCA.png"):
    """
    get coordinates of points of DCA curve
    """
    pt_arr = []
    net_bnf_arr = []
    y_pred = y_pred.ravel()
    for i in range(0,100,1):
        pt = i /100
        #compute TP FP
        y_pred_clip = (y_pred>pt).astype(int)
        TP = np.sum( y_true*np.round(y_pred_clip) )
        FP = np.sum((1 - y_true) * np.round(y_pred_clip))
        net_bnf = ( TP-FP* pt/(1-pt) )/len(y_true)
        pt_arr.append(pt)
        net_bnf_arr.append(net_bnf)
    
    return pt_arr, net_bnf_arr


y_true = y_test

fig = plt.figure(dpi=300)
y_proba = gbm_model.predict_proba(X_test)[:,1]  ##
pt_arr, net_bnf_arr = get_DCA_coords(y_proba,y_test)
plt.plot(pt_arr, net_bnf_arr, lw=2, linestyle='-',label='GBM') ##

plt.plot(pt_arr, np.zeros(len(pt_arr)), color='k', lw=1, linestyle='--',label='All negative')

pt_np = np.array(pt_arr)
pr = np.sum(y_true)/len(y_true)# Prevalence
all_pos = pr-(1-pr)*pt_np/(1-pt_np)
plt.plot(pt_arr, all_pos, color='b', lw=1, linestyle='dotted',label='All positive')

plt.xlim([0.0, 1.0])
plt.ylim([-1, 1])
plt.xlabel('Probability Threshold')
plt.ylabel('Net Benefit')
plt.title('DCA of GBM') ##
plt.savefig("./DCA of GBM.pdf") ##
plt.show()


# ## CatBoost

# In[145]:


from catboost import CatBoostClassifier
catb = CatBoostClassifier()
catb_model = catb.fit(X_train, y_train)
catb_model.score(X_train, y_train),catb_model.score(X_test, y_test)


# In[146]:


#交叉验证
scores = cross_val_score(catb_model, X_train, y_train, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[147]:


from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(catb_model, X_train, y_train, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())


# In[148]:


scores['train_prec_macro'].mean(), scores['train_prec_macro'].std()


# In[149]:


scores['train_rec_macro'].mean(),scores['train_rec_macro'].std()


# In[150]:


scores_f1 = cross_val_score(catb_model, X_train, y_train, cv=5, scoring='f1_macro')
scores_f1.mean(), scores_f1.std()


# In[151]:


#测试集交叉验证
scores = cross_val_score(catb_model, X_test, y_test, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[152]:


scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(catb_model, X_test, y_test, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())


# In[153]:


scores['test_prec_macro'].mean(), scores['test_prec_macro'].std()


# In[154]:


scores['test_rec_macro'].mean(),scores['test_rec_macro'].std()


# In[155]:


scores_f1 = cross_val_score(catb_model, X_test, y_test, cv=5, scoring='f1_macro')
scores_f1.mean(), scores_f1.std()


# In[156]:


from sklearn.metrics import plot_confusion_matrix

# disp = plot_confusion_matrix(clf, X_test, y_test, values_format='d')
disp = plot_confusion_matrix(catb_model, X_test, y_test, display_labels=['Live','Die'])  #None,'true','pred','all' 

disp.ax_.set_title('confusion matrix of Catboost',fontsize=20)
disp.ax_.set_xlabel('label predicted by  Catboost',fontsize=15)
disp.ax_.set_ylabel('true label',fontsize=15)

# # # 修改图标、字体大小等属性
# # fig,ax = plt.subplots(dpi=60)
# # disp = plot_confusion_matrix(clf, X_test, y_test, normalize=None, display_labels=['benign','malignant'],ax=ax)  #None,'true','pred','all' 
# # disp.ax_.set_title('confusion matrix of LogisticRegression',fontsize=20)
# # disp.ax_.set_xlabel('label predicted by  LogisticRegression',fontsize=20)
# # plt.show()

plt.savefig('./混淆矩阵_Catboost.pdf')
plt.show()


# In[157]:


y_pred = catb_model.predict(X_test)
#CatBost打印分类报告
print(classification_report(y_test, y_pred))


# In[158]:


## ROC
import matplotlib.pyplot as plt  
# from sklearn import datasets, metrics, model_selection, svm
from sklearn.metrics import plot_roc_curve

f,ax = plt.subplots(dpi=1200)
plot_roc_curve(catb_model , X_test, y_test, ax=ax, name='test roc')  ##
plot_roc_curve(catb_model , X_train, y_train, ax=ax, name='train roc')  ##
plt.plot([0,1],[0,1],":",label='by chance')
plt.legend()
plt.title('ROC on test set of CatBoost',fontsize=14)  ##
plt.savefig("./roc on test set of CatBoost.pdf")
plt.show() 


# In[159]:


# k折交叉验证的多条roc曲线画在同一张图上
from sklearn import metrics
from sklearn.metrics import roc_curve,auc

import numpy as np
# 数据
#X_ = scaler.transform(X_imputed)#标准化
#X_ = selector.transform(X_)#特征筛选

# 交叉验证及画roc图
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(dpi=300)
for i, (train_index, test_index) in enumerate(skf.split(X_, y)):
    catb_model.fit(X_[train_index], y[train_index])# 训练模型  ##
    viz = metrics.plot_roc_curve(catb_model, X_[test_index], y[test_index],  ##
                         name='ROC fold {}'.format(i),
                         alpha=0.5, lw=1, ax=ax)#画roc曲线
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

# # # ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
# # #         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
        lw=2, alpha=0.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="ROC Curve of StratifiedKFold for CatBoost")  ##
ax.legend(loc="lower right")
plt.grid()
plt.savefig("./roc of stratifiedKFold for CatBoost.pdf")  ##
plt.show()


# In[160]:


# 决策曲线分析图（DCA）
import numpy as np
def get_DCA_coords(y_pred,y_true,fn="DCA.png"):
    """
    get coordinates of points of DCA curve
    """
    pt_arr = []
    net_bnf_arr = []
    y_pred = y_pred.ravel()
    for i in range(0,100,1):
        pt = i /100
        #compute TP FP
        y_pred_clip = (y_pred>pt).astype(int)
        TP = np.sum( y_true*np.round(y_pred_clip) )
        FP = np.sum((1 - y_true) * np.round(y_pred_clip))
        net_bnf = ( TP-FP* pt/(1-pt) )/len(y_true)
        pt_arr.append(pt)
        net_bnf_arr.append(net_bnf)
    
    return pt_arr, net_bnf_arr


y_true = y_test

fig = plt.figure(dpi=300)
y_proba = catb_model.predict_proba(X_test)[:,1]  ##
pt_arr, net_bnf_arr = get_DCA_coords(y_proba,y_test)
plt.plot(pt_arr, net_bnf_arr, lw=2, linestyle='-',label='CatBoost') ##

plt.plot(pt_arr, np.zeros(len(pt_arr)), color='k', lw=1, linestyle='--',label='All negative')

pt_np = np.array(pt_arr)
pr = np.sum(y_true)/len(y_true)# Prevalence
all_pos = pr-(1-pr)*pt_np/(1-pt_np)
plt.plot(pt_arr, all_pos, color='b', lw=1, linestyle='dotted',label='All positive')

plt.xlim([0.0, 1.0])
plt.ylim([-1, 1])
plt.xlabel('Probability Threshold')
plt.ylabel('Net Benefit')
plt.title('DCA of CatBoost') ##
plt.savefig("./DCA of CatBoost.pdf") ##
plt.show()


# # 不同ML算法预测准确性比较

# In[161]:


models = [
    loj_model,
    nb_model,
    knn_model,
    svm_model,
    mlpc,
    cart_model,
    rf_model,
    gbm_model,
    #xgb_model,
    #lgbm_model,
    catb_model
]

for model in models:
    names = model.__class__.__name__
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("-"*28)
    print(names + " : " )
    print("Accuracy: {:.2%}".format(accuracy))

y_pred_mlpc = mlpc.predict(X_test_scaled)
accuracy_mlpc = accuracy_score(y_test, y_pred_mlpc)
print("-"*28)
print(mlpc.__class__.__name__ + " : " )
print("Accuracy: {:.2%}".format(accuracy_mlpc))
result = []
results = pd.DataFrame(columns= ["Models", "Accuracy"])
#以上将各算法对应准确性整理在数据框中
for model in models:
    names = model.__class__.__name__
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    result = pd.DataFrame([[names, accuracy*100]], columns= ["Models", "Accuracy"])
    results = results.append(result)
y_pred_mlpc = mlpc.predict(X_test_scaled)
accuracy_mlpc = accuracy_score(y_test, y_pred_mlpc)
result = pd.DataFrame([[mlpc.__class__.__name__ , accuracy_mlpc*100]], columns= ["Models", "Accuracy"])
results = results.append(result)
#条形图可视化
plt.figure(figsize=(16,12))
a=sbn.barplot(x = "Accuracy", y = "Models", data = results,palette=("Set1"))
plt.xlabel("Accuracy %")
plt.ylabel("Models Accuracy Score")
plt.xticks(fontproperties='Times New Roman',fontsize=12) #x轴刻度的字体大小
plt.yticks(fontproperties='Times New Roman',fontsize=12) #y轴刻度的字体大小
#plt.savefig('./不同机器学习算法的准确性比较.tiff', dpi=300)#输出
plt.savefig('./不同机器学习算法的准确性比较.pdf')#输出
plt.show()


# In[162]:


# 校准曲线
from sklearn.datasets import make_classification as mc
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB # 导入高斯朴素贝叶斯
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import brier_score_loss # 导入布里尔分数
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve # 对概率类模型进行校准，方法是分箱







name = ["Logistic","KNN","SVM","ANN","Decision Tree","Random Forest","GBM","Catboost"]  #



#开始画图
fig,ax1 = plt.subplots(figsize = (8,6))
ax1.plot([0,1],[0,1],label = "Perfectly calibrated") # 绘制对角线，把（0，0），（1，1）连起来
for clf,name_ in zip([loj_model,knn_model,svm_model,mlpc,cart_model,rf_model,gbm_model,catb_model],name):    ##
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    if hasattr(clf,"predict_proba"): #对象里如果有这个接口
        prob_pos = clf.predict_proba(X_test)[:,1]
    else: #就是针对SVM
        prob_pos = clf.decision_function(X_test)
        prob_pos = (prob_pos-prob_pos.min())/(prob_pos.max()-prob_pos.min())  #手动归一化
    clf_score = brier_score_loss(y_test,prob_pos,pos_label=y.max())
    #对只有0，1的标签值进行分箱后才能画图
    trueproba, predproba = calibration_curve(y_test, prob_pos, n_bins=3)
    ax1.plot(predproba,trueproba,"s-",label = "%s(%1.3f)"%(name_,clf_score))
    ax1.set_ylabel("True probability for class 1")
    ax1.set_xlabel("Mean Predcited probability")
    ax1.set_ylim([-0.05,1.05])
    ax1.legend()
plt.title('Calibration curve of models') ##
plt.savefig("./Calibration curve of models.pdf") ##
plt.show()


# In[163]:


###定义columns
feature_names = X.columns[np.where(selector.get_support())]


# In[164]:


#特征重要性评价整理为数据框
Importance = pd.DataFrame({"Importance": catb_model.feature_importances_},
                          index = feature_names)
#降序排列并画柱状图

Importance.sort_values(by = "Importance",
                       axis = 0,
                       ascending = True).plot(kind = "barh", color = "darkorange")
plt.xlabel("Variables Importance Ratio")
plt.yticks(fontproperties='Times New Roman',fontsize=4,rotation=40) #y轴刻度的字体大小、旋转角度（逆时针）

#plt.savefig('./随机森林重要特征.tiff', dpi=300, bbox_inches='tight')
plt.savefig('./catboost重要特征.pdf')
plt.show()


# # SHAP可视化

# In[165]:


import shap
shap.initjs()


# In[166]:


explainer = shap.TreeExplainer(catb_model) 


# In[167]:


X_trained_1 = pd.DataFrame(columns=feature_names, data=X_train)
X_tested_1 = pd.DataFrame(columns=feature_names, data=X_test)


# In[168]:


shap_values = explainer.shap_values(X_trained_1)


# In[169]:


shap_values2 = explainer(X_trained_1) 


# In[170]:


fig, ax = plt.subplots()
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#shap.dependence_plot("等效渗透率", shap_values, data_with_name,show=False)
shap.summary_plot(shap_values, X_trained_1,show=False)
plt.xticks( fontproperties='Times New Roman', size=12) #设置x坐标字体和大小
plt.yticks(fontproperties='Times New Roman', size=12) #设置y坐标字体和大小
plt.xlabel('SHAP value (impact on model output)', fontsize=10)#设置x轴标签和大小
plt.tight_layout() #让坐标充分显示，如果没有这一行，坐标可能显示不全
plt.savefig("全局特征排序-区分特征值_Catboost.pdf",dpi=1200) #可以保存图片


# In[171]:


#LN.metastasis.rate.分布与SHAP值的关系
fig, ax = plt.subplots()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#shap.dependence_plot("等效渗透率", shap_values, data_with_name,show=False)
shap.plots.scatter(shap_values2[:,'Tstage'],show=False,color='seagreen')
plt.xticks(fontproperties='Times New Roman', size=20) #设置x坐标字体和大小
plt.yticks(fontproperties='Times New Roman', size=20) #设置y坐标字体和大小
plt.xlabel('Tstage', fontsize=20)#设置x轴标签和大小
plt.tight_layout() #让坐标充分显示，如果没有这一行，坐标可能显示不全
plt.savefig("Tstage分布与SHAP值的关系_Catboost.pdf",dpi=1200) #可以保存图片
#plt.show


# In[172]:


#第一个样本的力图
fig, ax = plt.subplots()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#shap.dependence_plot("等效渗透率", shap_values, data_with_name,show=False)
shap.force_plot(explainer.expected_value, shap_values[0], 
                feature_names= list(feature_names), show = False, matplotlib = True)
plt.xticks( fontproperties='Times New Roman', size=20) #设置x坐标字体和大小
plt.yticks(fontproperties='Times New Roman', size=20) #设置y坐标字体和大小
plt.xlabel('First sample', fontsize=20)#设置x轴标签和大小
plt.tight_layout() #让坐标充分显示，如果没有这一行，坐标可能显示不全
plt.savefig("第一个样本的力图.pdf",dpi=1200) #可以保存图片
#plt.show


# In[173]:


#第二个样本的力图
fig, ax = plt.subplots()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#shap.dependence_plot("等效渗透率", shap_values, data_with_name,show=False)
shap.force_plot(explainer.expected_value, shap_values[1], 
                feature_names= list(feature_names),show = False, matplotlib = True)
plt.xticks( fontproperties='Times New Roman', size=20) #设置x坐标字体和大小
plt.yticks(fontproperties='Times New Roman', size=20) #设置y坐标字体和大小
plt.xlabel('Second sample', fontsize=20)#设置x轴标签和大小
plt.tight_layout() #让坐标充分显示，如果没有这一行，坐标可能显示不全
plt.savefig("第二个样本的力图.pdf",dpi=1200) #可以保存图片
#plt.show


# In[174]:


#第二个样本的力图
fig, ax = plt.subplots()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#shap.dependence_plot("等效渗透率", shap_values, data_with_name,show=False)
shap.force_plot(explainer.expected_value, shap_values[2], 
                feature_names= list(feature_names),show = False, matplotlib = True)
plt.xticks( fontproperties='Times New Roman', size=20) #设置x坐标字体和大小
plt.yticks(fontproperties='Times New Roman', size=20) #设置y坐标字体和大小
plt.xlabel('Second sample', fontsize=20)#设置x轴标签和大小
plt.tight_layout() #让坐标充分显示，如果没有这一行，坐标可能显示不全
plt.savefig("第三个样本的力图.pdf",dpi=1200) #可以保存图片
#plt.show


# In[ ]:




