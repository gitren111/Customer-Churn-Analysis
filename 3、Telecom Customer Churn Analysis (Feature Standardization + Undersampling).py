#一、模块导入
#1、数据处理
import pandas as pd
import numpy as np
#2、可视化
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid',font_scale=1.3)
plt.rcParams['font.family']='SimHei'
plt.rcParams['axes.unicode_minus']=False
#3、统计分析
from scipy.stats import chi2_contingency#卡方检验
from scipy.stats import levene#齐性检验的Levene’s检验
from scipy.stats import f_oneway#单因素方差的分析
#4、特征工程
import sklearn
from sklearn import preprocessing#数据预处理
from sklearn.preprocessing import LabelEncoder#编码转换
from sklearn.preprocessing import StandardScaler#标准归一化
from sklearn.model_selection import StratifiedShuffleSplit#分层抽样
from imblearn.over_sampling import SMOTE#过抽样SMOTE方法增加少数类样本
from imblearn.under_sampling import RandomUnderSampler#欠抽样
from sklearn.model_selection import train_test_split#训练和测试数据分区
from sklearn.decomposition import PCA#主成分分析（降维）
#5、分类算法
from sklearn.ensemble import RandomForestClassifier#随机森林
from sklearn.svm import SVC,LinearSVC#支持向量机
from sklearn.linear_model import LogisticRegression#逻辑回归
from sklearn.neighbors import KNeighborsClassifier# KNN算法
from sklearn.cluster import KMeans#kmeans聚类算法
from sklearn.naive_bayes import GaussianNB#朴素贝叶斯
from sklearn.tree import  DecisionTreeClassifier#决策树
#6、分类算法-集成学习
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
#7、模型评估
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import VotingClassifier
#8、忽略警告
import warnings
warnings.filterwarnings('ignore')

#二、数据清洗
#1、读取数据
df = pd.read_csv(r'WA_Fn-UseC_-Telco-Customer-Churn.csv',header=0)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
print(df.head())
print(df.shape)

#2、数据清洗
#(1)查看缺失值
print('缺失值\n',df.isnull().sum())
#(2)查看重复值
print('重复值\n',df.duplicated().sum())
#（3）查看数据类型
print('数据类型',df.info())
"""
TotalCharges这一列数据类型是object，应该改为float64数据类型
"""
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')
print(df['TotalCharges'].dtype)#查看转换后的数据类型是不是float64
#（4）转换数据后再次查看缺失值
print('缺失值第二次查询\n',df.isnull().sum())#TotalCharges有11个缺失值
"""
缺失值填充的原则：
分类型数据：众数填充(因为不是数值型，用出现次数最多类别填充，保持数据原有分布的特性）
数值型数据：正态分布，均值/中位数填充（保持数据的对称性）；偏态分布，中位数填充（中位数比均值更能代表数据的中心趋势，减少极端值对整体数据的影响）。
"""
#（5）做直方图查看数据分布状态，再决定填充方式：全部客户、流失客户、留存客户分别做图
plt.figure(figsize=(14,5))
plt.subplot(1,3,1)
plt.title('全部客户的总付费直方图')
sns.distplot(df['TotalCharges'].dropna())

plt.subplot(1,3,2)
plt.title('流失客户的总付费直方图')
sns.distplot(df[df['Churn']=='Yes']['TotalCharges'].dropna())

plt.subplot(1,3,3)
plt.title('留存客户的总付费直方图')
sns.distplot(df[df['Churn']=='No']['TotalCharges'].dropna())
plt.show()

#(6)三个直方图均显示未偏态分布，选择中位数填充
df.fillna({'TotalCharges':df['TotalCharges'].median()},inplace=True)
print('填充中位数后检查空值\n',df.isnull().sum())

#3、查看样本分布
#(1)将'churn'列重新编码为0 1
df['Churn']=df['Churn'].map({'Yes':1,'No':0})
print('预览重新编码后的Churn',df['Churn'].head())#检查
#(2)绘制饼图查看流失客户占比
churn_value = df['Churn'].value_counts()
labels = df['Churn'].value_counts().index
churn_per = churn_value[1]/churn_value.sum()*100
plt.figure(figsize=(7,7))
plt.pie(churn_value,labels=labels,colors=['b','w'],explode=(0.1,0),autopct='%1.1f%%',shadow=True)
plt.title('流失客户占比:{:.1f}%'.format(churn_per))
plt.show()

#三、特征选择
#1、整数编码
features = df.iloc[:,1:20]#提取特征
#（1）查看变量间两两相关性
corr_df = features.apply(lambda x: pd.factorize(x)[0])
print('特征整数编码\n',corr_df.head())
corr = corr_df.corr()
print('特征相关性矩阵\n',corr)
#（2）可视化相关性矩阵：画热力图
plt.figure(figsize=(17,14))
ax = sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,
                 linewidths=0.2,cmap='RdYlGn',annot=True)
plt.title('特征相关性')
plt.show()
"""
观察热力图可知：InternetService、OnlineSecurity、OnlineBackup、DeviceProtection、TechSupport、StreamingTV
StreamingMovies之间相关性很强，PhoneService、MultipleLines业务也存在很强正相关关系
"""

#2、独热编码onehot：查看研究对象“churn”用户流失与其他特征的相关性
df_onehot = pd.get_dummies(df.iloc[:,1:21])
#(1)可视化：查看用户流失churn与各个特征关系
plt.figure(figsize=(15,12))
df_onehot.corr()['Churn'].sort_values(ascending=False).plot(kind='bar')
plt.title('用户流失与其他特征的相关性')
print(df_onehot.corr()['Churn'].sort_values(ascending=False))
plt.show()
"""
相关性分析仅提供变量之间线性关系的度量，并不意味着因果关系。即使相关性较弱，某些特征在模型中也可能是有用的，
观察图可知：
相关大于15%：Contract、OnlineSecurity、TechSupport、InternetService、PaymentMethod、OnlineBackup、DeviceProtection
SeniorCitizen、Partner、Dependents、MonthlyCharges、tenure、StreamingTV、StreamingMovies、PaperlessBilling、TotalCharges
0相关几乎为0：gender、PhoneService
"""
#3、与churn用户流失相关性大于15%的特征合并并分析:这里将相关小的剔除，后面模型里面也不再考虑
kf_var = list(df.columns[2:6])
for var in list(df.columns[8:20]):
    kf_var.append(var)
print('kf_var=',kf_var)

#四、统计分析
#1、频数分布比较：卡方检验（针对分类变量）
"""
每组间数据有显著性差异，频数分布（对频数进行比较，以了解数据分布的差异或相似性）比较才有意义,否则可能会做无用功。
“卡方检验”可以判断组间是否有显著性差异，决定是否继续进行频数分布比较（主要用于检验两个分类变量之间是否独立，而无法检验连续变量）
"""
def KF(x):
    df1 = pd.crosstab(df['Churn'],df[x])
    li1 = list(df1.iloc[0,:])
    li2 = list(df1.iloc[1,:])
    kf_data = np.array([li1,li2])
    #将两个列表 li1 和 li2 作为两个单独的数组元素，创建了一个新的二维 NumPy 数组进行检验两个变量是否独立
    kf = chi2_contingency(kf_data)
    if kf[1] < 0.05:
        print('churn by {} 的卡方临界值是{:.2f},小于0.05，说明{}组间有显著性差异，可进行【交叉分析】'.format(x,kf[1],x),'\n')
    else:
        print('churn by {} 的卡方临界值是{:.2f},大于0.05，表面{}组间无显著性差异，不可进行交叉分析'.format(x,kf[1],x),'\n')
    """
    检验结果kf有四个值：
    1、chi2：卡方统计量的值
    2、p：P 值，表示卡方统计量在原假设（变量之间独立）下的概率分布，通常小于显著性水平，如 0.05，拒绝原假设有显著关系
    3、dof：自由度
    4、expected：在原假设成立的情况下，期望频数的矩阵
    """
print('kf_var高相关系特征与用户流失的卡方检验结果如下：','\n')
print(list(map(KF,kf_var)))
"""卡方检验发现TotalCharges组间无显著性差异，因为TotalCharges是连续变量，放到后面专门测试"""

#2、频数分布比较：柱状图
plt.figure(figsize=(20,25))
a = 0
for k in [var for var in kf_var if var != 'tenure' and var != 'MonthlyCharges' and var != 'PaymentMethod'
          and var != 'TotalCharges']:
    #因为TotalCharges、tenure、MonthlyCharges为连续变量（数值），用柱状图无法好的分类，PaymentMethod单独画效果好
    a = a+1
    plt.subplot(3,4,a)
    plt.title('churn by ' + k,fontsize=10)
    sns.countplot(x=k,hue='Churn',data=df)
    plt.tick_params(axis='x',labelsize=8)
plt.subplots_adjust(wspace=0.3,hspace=0.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,8))
plt.title('churn by PaymentMethod')
sns.countplot(x='PaymentMethod',hue='Churn',data=df)
plt.show()
"""
可以通过柱状图看出具体特征类别对客户流失的影响，但是还不能就此判断，因为样本不均衡（流失客户样本占比26.5%）
所以不能直接通过频数的柱状图去分析，用百分比可以更清楚了解流失客户在总样本的比例，用交叉表分析不同特征变量与客户流失的关系
解决办法：交叉分析切做同行百分比（’churn作为行‘）
"""

#3、交叉分析
print('ka_var列表中的特征向量与Churn用户流失交叉分析结果如下：','\n')
for i in [var for var in kf_var if var != 'MonthlyCharges' and var != 'tenure'
          and var != 'TotalCharges']:
    print('...............churn by {}...............'.format(i))
    print(pd.crosstab(df['Churn'],df[i],normalize=0),'\n')#交叉分析,同行百分比
    """
    交叉分析是针对分类变量进行的，因为它的目的是查看不同类别之间的分布差异。对于连续变量，通常需要进行分箱操作，
    将连续变量转换为分类变量，然后再进行交叉分析,MonthlyCharges 和 tenure不分箱就不要做交叉分析，因为会得到很多组，使得难以解释
    """


#4、均值比较：组间有显著性差异，均值比较才有意义（针对连续变量）
"""
显著性检验后，先通过齐性分析，再通过方差分析，最后才能做均值比较
1、齐性分析:确定参与方差分析的多组数据是否来自具有相同方差的总体。如果各组数据的方差不相等，那么这些数据被认为是不具有方差齐性的。
p值大于常用的显著性水平，如0.05,则可以认为方差是齐的
2、方差分析（ANOVA）：主要用于连续变量（前面漏的特征），用于比较两个或多个样本组的均值是否存在显著差异。基本思想是比较组间变异和组内变异，以判断这些样本组是否来自于具有相同均值的总体。
如果p值小于显著性水平（例如0.05），我们拒绝零假设，认为至少有一个样本组的均值与其他样本组不同。
组间变异：不同样本组之间均值差异所引起的变异
组内变异：每个样本组内部观测值与其组均值之间的差异
F 统计量：组间均方/组内均方
"""
def ANOVA(x):
    li_index = list(df['Churn'].value_counts().keys())
    args = []
    for i in li_index:
        args.append(df[df['Churn']==i][x])
    w,p = levene(*args)
    if p < 0.05:
        print('警告：churn by {}的p值是{:.2f},小于0.05，表明齐性检验不通过，不可做方差分析'.format(x,p),'\n')
    else:
        f,p_value = f_oneway(*args)
        print('churn by {} 的f值是{},p值是{}'.format(x,f,p_value),'\n')
        if p_value < 0.05:
            print('churn by {}的均值有显著性差异，可进行均值比较'.format(x),'\n')
        else:
            print('churn by {}的均值无显著差异，不可进行均值比较'.format(x),'\n')
"""
对连续变量TotalCharges、MonthlyCharges、tenure进行齐性检验和方差分析
结果：三个变量均不通过齐性检验，不可进行方差分析
"""
print('TotalCharges、MonthlyCharges、tenure的齐性检验和方差分析结果如下：','\n')
ANOVA('TotalCharges')
ANOVA('MonthlyCharges')
ANOVA('tenure')
"""
总结-分析结果：
1、SeniorCitizen：年轻客户在流失和留存占比都高
2、Partner：单身客户易流失
3、Dependents：经济不独立易流失
4、InternetService：办理Fiber optic光纤易流失
5、OnlineSecurity：没开通网络安全易流失
6、OnlineBackup：没开通网络备份易流失
7、DeviceProtection：没开通设备保护服务易流失
8、TechSupport：没开通技术支持易流失
9、StreamingTV、StreamingMovies：开通网络电视和电影对用户留存和流失影响不明显
10、Contract：逐月签合同易流失
11、PaperlessBilling：用电子账单易流失
12、PaymentMethod：用电子支票支付更容易流失
我们可以在SQL上找有以上特征的客户，进行精准营销，即可以降低用户流失。虽然特征选得越多，越精确，但覆盖的人群也会越少。
故我们还需要计算“特征”的【重要性】，将最为重要的几个特征作为筛选条件。
"""

#五、特征工程：计算特征的【重要性】，是“分类视角”，挑选常见的分类模型，进行批量训练，然后挑出得分最高的模型，进一步计算“特征重要性”。
#1、提取特征
"""
由上一轮分析可将gender、PhoneService剔除，customerID是随机数不影响建模也剔除
"""
churn_var = df.iloc[:,2:20]
churn_var.drop('PhoneService',axis=1,inplace=True)
print(churn_var.head())
print(churn_var.dtypes)
#(1)判断量纲差异大小：最小最大值
numeric_cols = churn_var.select_dtypes(include=[np.int64,np.float64])
max_v = numeric_cols.max()
min_v = numeric_cols.min()
print('特征列的最大值')
print(max_v)
print('特征列的最小值')
print(min_v)
value_ranges = max_v-min_v
print('特征列的取值范围')
print(value_ranges)
"""
根据结果可知：SeniorCitizen是个二分类问题，只有tenure、MonthlyCharges、TotalCharges存在量纲差异大问题
解决思路：标准化、离散化，哪个模型精度高选哪个
"""
#2、处理量纲差异大（连续变量）
#（1）标准化
scaler = StandardScaler(copy=False)
scaler.fit_transform(churn_var[['tenure','MonthlyCharges','TotalCharges']])
#fit_transform先进行每个特征均值和标准差计算，然后对每个数据进行标准化处理x-u/o
churn_var[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(churn_var[['tenure',
                                                                                    'MonthlyCharges',
                                                                                    'TotalCharges']])
#将转换的标准化数据赋值给原列表
print(churn_var[['tenure','MonthlyCharges','TotalCharges']].head())

#3、分类数据转换为整数编码
#(1)查看分类变量的标签：SeniorCitizen实际只有0 1也是分类变量也加进去
def label(x):
    print(x,'--',churn_var[x].unique())
df_object = churn_var.select_dtypes(include='object')
df_object['SeniorCitizen'] = churn_var['SeniorCitizen']
print(df_object.dtypes)
print(list(map(label,df_object)))
"""
分析：通过前面交叉分析发现，特征OnlineSecurity、OnlineBackup、DeviceProtection、TechSupport、StreamingTV、StreamingMovies
下面的标签No internet service都是一样的，这6个特征都跟网上增值服务有关，可以判断No internet service不影响客户流失
逻辑是这6项增值服务必须要开通互联网服务才能享受，没开通互联网服务就视为没开通这6项增值服务，所以将这6个特征中的No internet service标签
合并到no里面
"""
churn_var.replace(to_replace='No internet service',value='No',inplace=True)
print('调整分类特征标签后\n',list(map(label,df_object)))

#(2)整数编码
def Label_encode(x):
    churn_var[x] = LabelEncoder().fit_transform(churn_var[x])
for i in range(0,len(df_object.columns)):
    Label_encode(df_object.columns[i])
print(list(map(label,df_object.columns)))

#4、处理样本不均衡：取精度最好的抽样方式
"""
一、判断样本是否均衡：流失客户比例小于5%或者流失/留存客户小于20%
1、流失26.5%/流存73.5%=36%，属于不完全均衡
二、处理不均衡常用方式
1、分层抽样：概率抽样技术，它在每个层次内保持原始数据集中的类别比例。
优点：保持了原始数据集的类别分布；减少了抽样偏差，提高了模型的泛化能力。
缺点：需要更多的数据来保持分层。
2、过抽样：增加少数类的样本数量，使其与多数类的样本数量相等或接近。
优点：可以增加少数类的代表性，有助于模型学习少数类的特征；
缺点：如果简单地复制样本，可能会导致过拟合。合成样本可能不反映真实数据的分布。
3、欠抽样：减少多数类的样本数量，使其与少数类的样本数量相等或接近。
优点：可以增加少数类的代表性，有助于模型学习少数类的特征；可以减少模型的训练时间。
缺点：可能会丢失重要的信息，因为删除了部分数据。
"""
#（1）欠抽样
x = churn_var
y = df['Churn']
rus = RandomUnderSampler(sampling_strategy='auto',random_state=0)#初始化欠抽样器，随机选择少数类样本并减少多数类样本
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
#欠抽样要先划分训练和测试数据集，确保测试数据集分布不变，因为欠抽样只对训练集进行操作（过抽样是对整个数据集进行操作，两个划分顺序不同）
x,y = rus.fit_resample(x_train,y_train)#欠抽样指定对训练集进行
print('欠抽样后训练集大小: {}, 原测试集大小: {}'.format(len(x),len(x_test)))
print('欠抽样后训练标签集大小: {}, 原测试标签集大小: {}'.format(len(y),len(y_test)))
print('过抽样后训练集标签样本',pd.Series(y).value_counts())#训练集样本留存和流失客户变成1:1，测试集不变


#六、数据建模：
#1、分类算法
Classifiers = [['Random Forest',RandomForestClassifier()],
               ['Support Vector Machine',SVC()],
               ['LogisticRegression',LogisticRegression()],
               ['Naive Bayes',GaussianNB()],
               ['Decision Tree',DecisionTreeClassifier()],
               ['AdaBoostClassifier',AdaBoostClassifier()],
               ['GradientBoostingClassifier',GradientBoostingClassifier()],
               ['XGB',XGBClassifier()],
               ['CatBoosting',CatBoostClassifier(logging_level='Silent')]]

#2、训练模型
classify_result = []
names = []
predictions = []
for name,classifier in Classifiers:
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    recall = recall_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    f1score = f1_score(y_test,y_pred)
    class_eva = pd.DataFrame([recall,precision,f1score])
    classify_result.append(class_eva)
    name = pd.Series(name)
    names.append(name)
    y_pred = pd.Series(y_pred)
    predictions.append(y_pred)

#七、模型评估与对比
names = pd.DataFrame(names)
names = names[0].tolist()#将名称转换成列表
result = pd.concat(classify_result,axis=1)#按列拼接
result.columns = names#行设置为分类器名称
result.index = ['recall','precision','f1score']
print(result)
"""
结论：6个不同的量纲差异和样本不均衡处理方式比较：
一、得分比较
1、特征工程采取标准化处理量纲差异，采取过抽样处理样本不均衡，随机森林算法最终模型得分最高0.844
2、特征工程采取离散化处理量纲差异，采取过抽样处理样本不均衡，也是随机森林算法最终模型得分最高0.831
二、基于随机森林模型输出特征重要性
"""
#八、输出特征重要性
#1.1、基于“Random Forest”模型输出特征重要性
rf = RandomForestClassifier(n_estimators=100,random_state=0)#默认100颗数
rf.fit(x_train,y_train)
importances = rf.feature_importances_#获取特征重要性
indices = np.argsort(importances)[::-1]
#np.argsort返回的是数据排列后的索引（默认升序），降序排列[::-1] 表示从后向前，步长为 -1
feature_names = x_train.columns
print('特征重要性',importances)
print('特征重要性索引',indices)
for f in range(x_train.shape[1]):
    print('{} : {}'.format(x_train.columns[indices[f]],importances[indices[f]]))

#可视化
plt.figure(figsize=(10,8))
plt.title('Random Forest模型特征重要性')
plt.bar(range(len(importances)),importances[indices],align='center')
plt.xticks(range(len(importances)),feature_names[indices],rotation=90)
plt.xlim(-1,x_train.shape[1])
plt.tight_layout()
plt.show()

#2、特征分析：由于Random Forest得分最高，故以Random Forest得到的特征重要性进行分析
#（1）第一重要特征：MonthlyCharges
"""
思路：MonthlyCharges和TotalCharges存在天然关系，而且分别是第一和第三重要特征，将连续变量按特点区间分组
然后进行卡方检验-频数分布比较
"""
print('第一和第四特征分析',df[['MonthlyCharges','TotalCharges']].describe())#先了解分布情况
#数据分箱
M_bins = [18.25,36,71,90,118.75]
T_bins = [18.8,403,1398,3787,8684.8]
mc = pd.cut(df['MonthlyCharges'],M_bins,labels=[1,2,3,4],right=False)
tc = pd.cut(df['TotalCharges'],M_bins,labels=[1,2,3,4],right=False)
df['MonthlyCharges_kf'] = mc
df['TotalCharges_kf'] = tc
KF('MonthlyCharges_kf')
KF('TotalCharges_kf')
"""
MonthlyCharges_kf、TotalCharges_kf的P值都小于0.05，有显著性差异，继续进行交叉分析
"""
#交叉分析
for i in ['MonthlyCharges_kf','TotalCharges_kf']:
    print('...............churn by {}...............'.format(i))
    print(pd.crosstab(df['Churn'],df[i],normalize=0),'\n')
"""
分析：
月付费71-118.75元的客户更容易流失
年付费403-3787元的客户更容易流失
"""
#画四分图：流失客户
df_1 = df[df['Churn']==1]
df_0 = df[df['Churn']==0]
plt.figure(figsize=(10,10))
sns.scatterplot(x='MonthlyCharges',y='TotalCharges',hue='Churn',data=df_1)
plt.axhline(y=df['TotalCharges'].mean(),linestyle='-',c='k')
plt.axvline(x=df['MonthlyCharges'].mean(),linestyle='-',c='green')
plt.title('流失客户四分图')

#画四分图：留存客户
plt.figure(figsize=(10,10))
sns.scatterplot(x='MonthlyCharges',y='TotalCharges',hue='Churn',data=df_0)
plt.axhline(y=df['TotalCharges'].mean(),linestyle='-',c='k')
plt.axvline(x=df['MonthlyCharges'].mean(),linestyle='-',c='green')
plt.title('留存客户四分图')
plt.show()

#（2）第二重要特征：tenure
plt.figure(figsize=(10,20))
sns.countplot(x='tenure',hue='Churn',data=df)
plt.title('Churn by tenure')
plt.tight_layout()
plt.show()#公司合作1-5个月的客户最容易流失

#(3)第四重要特征：Contract
"""
前面已完成的分类特征向量与Churn用户流失交叉分析结果： Month-to-month合约更容易流失
"""
"""
最终结论：综合“统计分析”和“Random Forest算法输出特征重要性”，得出流失客户有以下特征（依特征重要性从大到小排序）
1、MonthlyCharges：月付费71-118.75元的客户更容易流失
2、tenure：公司合作1-5个月的客户最容易流失
3、Contract：Month-to-month按月签订合约方式更容易流失
4、TotalCharges：年付费403-3787元的客户更容易流失
5、PaymentMethod：用电子支票支付更容易流失
6、InternetService：办理Fiber optic光纤易流失
7、OnlineSecurity：没开通网络安全易流失
8、TechSupport：没开通技术支持易流失
9、PaperlessBilling：用电子账单易流失
10、Dependents：经济不独立易流失
11、Partner：单身客户易流失
12、OnlineBackup：没开通网络备份易流失
13、DeviceProtection：没开通设备保护服务易流失
14、SeniorCitizen：年轻客户在流失和留存占比都高
15、StreamingMovies、StreamingTV：开通网络电视和电影对用户留存和流失影响不明显
"""
