
import os
import pickle


'''
    function: 对数据进行压缩,压缩为二进制格式,节约空间
'''
def dump_pkl(vocab, pkl_path, overwrite=True):
    if not pkl_path:
        print("pkl_path is None")
        return

    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return

    if pkl_path:
        with open(pkl_path, mode="wb") as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.dump(vocab, f, protocol=0)
        print("save %s success!" % pkl_path)



def load_pkl(pkl_path):
    with open(pkl_path, mode="rb") as f:
        result = pickle.load(f)
    return result





