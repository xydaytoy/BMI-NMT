import math
from collections import defaultdict

threshold = 0.4

#count word in sentence
def count_tf(data):
    tf_char = defaultdict(int)
    for line in data:
        tmp=set()
        for char in line:
            if char.strip():
                if char not in tmp:
                    tmp.add(char)
                    tf_char[char] += 1
    return tf_char

#count word pair in sentence
def count_co(src, tgt, len):
    co_pair = defaultdict(int)
    for ss, tt in zip(src, tgt):
        tmp=set()
        for s in ss:
            for t in tt:
                if (s,t) not in tmp:
                    tmp.add((s,t))
                    co_pair[(s,t)] += 1
    return co_pair

if __name__ =="__main__":
    with open("../train.en.shuf", "r", encoding="utf-8") as src_data, \
            open("../train.de.shuf", "r", encoding="utf-8") as tgt_data, \
            open("../src_dict.txt", "w", encoding="utf-8") as sd, \
            open("../tgt_dict.txt", "w", encoding="utf-8") as td, \
            open("../pair_dict.txt", "w", encoding="utf-8") as pd:
        src_num = 0 # num of src sentence
        tgt_num = 0 # num of tgt sentence
        src = []
        tgt = []
        for src_line in src_data.readlines():
            src_num = src_num + 1
            src_line = src_line.strip().split()
            src.append(src_line)
        for tgt_line in tgt_data.readlines():
            tgt_num = tgt_num+1
            tgt_line = tgt_line.strip().split()
            tgt.append(tgt_line)
        assert src_num == tgt_num
        print("read finish.")

        src_dict = count_tf(src)
        tgt_dict = count_tf(tgt)
        co_dict = count_co(src,tgt,src_num)
        for idx,val in src_dict.items():
            sd.write("{} {}\n".format(idx,val))
        for idx,val in tgt_dict.items():
            td.write("{} {}\n".format(idx,val))
        for idx,val in co_dict.items():
            pd.write("{} {} {}\n".format(idx[0],idx[1],val))
        print("count finish.")
    with open("../mi.txt", "w", encoding="utf-8") as mi_out:
        mi_dict={}
        for s,t in zip(src, tgt):
            weight_list = []
            for word in t:
                weight = 0
                max_len = 0
                for pos in s:
                    if (pos,word) not in mi_dict:
                        tmp_mi=math.log(1+src_num*co_dict[(pos,word)]/(src_dict[pos]*tgt_dict[word]))
                        mi_dict[(pos,word)]=tmp_mi
                    else:
                        tmp_mi=mi_dict[(pos,word)]
                    weight += tmp_mi
                    max_len += 1
                weight=weight/max_len
                if weight > threshold:
                    weight_list.append(weight)
                else:
                    weight_list.append(-1.0)
            str_weight = ' '.join([str(i) for i in weight_list])
            mi_out.write("{}\n".format(weight))
    print("mi.txt finish")     
