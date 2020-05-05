'''
    生成用于训练词向量的数据sentences.txt
    利用gensim训练skip-gram模型中的词向量,并保存词向量
'''


from gensim.models.fasttext import FastText
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
from utils.data_utils import dump_pkl
import time


# 加载数据的路径
train_x_seg_path = "./data/train_set_seg_x.txt"
train_y_seg_path = "./data/train_set_seg_y.txt"
test_x_seg_path = "./data/test_set_seg_x.txt"

sentence_path = './data/sentences.txt'  #用于训练词向量的数据
w2v_bin_path = "./data/w2v.model"       #保存为model或bin文件都行
# w2v_bin_path = "./data/w2v.bin"
save_model_txt_path = "./data/word2vec.txt"


'''
    function: 返回一行一行的数据
    col_seq:分隔符
'''
def read_lines(path, col_seq=None):
    lines = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if col_seq:
                if col_seq in line:
                    lines.append(line)
            else:
                lines.append(line)

    return lines


'''
    function: 将训练集x数据, 训练集y数据, 测试集x数据组合在一起（为一个很大的句子）
'''
def extract_sentence(train_x_seg_path, train_y_seg_path, test_x_seg_path):
    ret = []
    lines = read_lines(train_x_seg_path)
    lines += read_lines(train_y_seg_path)
    lines += read_lines(test_x_seg_path)
    for line in lines:
        ret.append(line)
    return ret



def save_sentence(sentence, save_path):
    with open(save_path, mode="w", encoding="utf-8") as f:
        for line in sentence:
            f.write("%s" % line)
    f.close()


'''
    function: 训练词向量, 保存词向量,测试相似性
'''
# 注意,对于样本比较小的数据集,要将min_count的值设置小一点儿,否则会报
def build_skip_gram_model(train_x_seg_path, train_y_seg_path, test_x_seg_path,
                          w2v_bin_path, sentence_path="", min_count=1):
    '''
    使用gensim训练词向量
    :param train_x_seg_path: 训练集x路径
    :param train_y_seg_path: 训练集y路径
    :param test_x_seg_path: 测试集x路径
    :param w2v_bin_path: 保存训练模型的领
    :param sentence_path: 保存拼接的大句子的路径
    :param min_count: 词频阈值
    :return:
    '''
    sentence = extract_sentence(train_x_seg_path, train_y_seg_path, test_x_seg_path)
    save_sentence(sentence, sentence_path)

    # train skip-gram model的词向量
    print("train w2v model")
    #workers:线程数, size:词向量维度,iter:训练多少轮
    # 使用Word2Vec训练词向量
    w2v = Word2Vec(sg=1, sentences=LineSentence(sentence_path), workers=6, size=300, window=5, min_count=min_count, iter=5)

    # 使用FastText训练词向量
    # w2v = FastText(sg=1, sentences=LineSentence(sentence_path), workers=8, size=300, window=5, min_count=min_count, iter=1)

    # 注意：不同的保存方式对应不同的加载模型的方式
    w2v.save(w2v_bin_path)          #可以进行二次训练
    # w2v.wv.save(w2v_bin_path)     #占用存储空间更小,但是不能进行二次训练
    # w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)    #加载时要用KeyedVectors.load_word2vec_format方法加载模型
    print("save %s ok" % w2v_bin_path)


    # test
    sim = w2v.wv.similarity('技师', '车主')
    # sim = w2v.most_similar("技师")
    print('技师 vs 车主 similarity score:', sim)


def load_model(w2v_bin_path, save_txt_path):
    # load model（加载模型的方法）
    # 注意：不同的保存方式对应不同的加载模型的方式
    # skip_gram_model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    skip_gram_model = Word2Vec.load(w2v_bin_path)
    print(skip_gram_model.most_similar("车子"))
    word_dict = {}
    # 从模型中加载词向量
    for word in skip_gram_model.wv.vocab:
        word_dict[word] = skip_gram_model[word]

    # 将从模型中加载的数据进行压缩保存,保存为二进制文件,节约空间
    dump_pkl(word_dict, save_txt_path, overwrite=True)



if __name__ == "__main__":
    start_time = time.time()
    build_skip_gram_model(train_x_seg_path, train_y_seg_path, test_x_seg_path, w2v_bin_path, sentence_path)
    end_time = time.time()
    print("train model time: %d seconds" % (end_time - start_time))
    load_model(w2v_bin_path, save_model_txt_path)
