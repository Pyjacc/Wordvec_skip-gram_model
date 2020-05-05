# Wordvec_skip-gram_model
NLP学习，第一章homework

## Homework-week1
### 1. 熟悉项目数据
数据地址：https://aistudio.baidu.com/aistudio/competition/detail/3
### 2. 数据处理
对数据做预处理：如去除缺失数据、去除噪音词(无用词)汇或者特殊符号 (见clean_data.py)
### 3. 对数据进行分词处理
使用jieba分词进行中文分词处理
### 4. 建立Vovab词汇表
应用分词后的数据建立Vovab词汇表，并保存到vocab.txt  （见build_vocab_dict.py）
词汇表格式为：词 词的index
## Homework-week2
### 1. 通过gensim训练词向量
#### 1.1 利用分词后的数据生成用于训练词向量的训练数据  
即Homework-week1中的vocab.txt
#### 1.2 保存训练数据
#### 1.3 应用gensim中Word2Vec或Fasttext训练词向量  
见train_word2vec_model.py中的build_skip_gram_model函数
#### 1.4 保存训练好的词向量
### 2. 构建embedding_matrix
读取上一步计算的词向量和构建的vocab词表，以vocab中的index为key值构建embedding_matrix  
eg: embedding_matrix[i] = [embedding_vector]
