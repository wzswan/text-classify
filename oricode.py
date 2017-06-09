from gensim import corpora, models
from scipy.sparse import csr_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
import numpy as np
import os,re,time,logging
import jieba
import pickle as pkl

# logging.basicConfig(level=logging.WARNING,
#                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                     datefmt='%a, %d %b %Y %H:%M:%S',
#                     )


class loadFolders(object):   #
    def __init__(self,par_path):
        self.par_path = par_path
    def __iter__(self):
        for file in os.listdir(self.par_path):
            file_abspath = os.path.join(self.par_path, file)
            if os.path.isdir(file_abspath): # if file is a folder
                yield file_abspath
class loadFiles(object):
    def __init__(self,par_path):
        self.par_path = par_path
    def __iter__(self):
        folders = loadFolders(self.par_path)
        for folder in folders:              # level directory
            catg = folder.split(os.sep)[-1]
            for file in os.listdir(folder):     # secondary directory
                file_path = os.path.join(folder,file)
                if os.path.isfile(file_path):
                    this_file = open(file_path,'rb')
                    content = this_file.read().decode('utf8')
                    yield catg,content
                    this_file.close()

def convert_doc_to_wordlist(str_doc,cut_all):
    sent_list = str_doc.split('\n')
    sent_list = map(rm_char, sent_list) # \u3000
    word_2dlist = [rm_tokens(jieba.cut(part,cut_all=cut_all)) for part in sent_list] #
    word_list = sum(word_2dlist,[])
    return word_list
def rm_tokens(words): #
    words_list = list(words)
    stop_words = get_stop_words()
    for i in range(words_list.__len__())[::-1]:
        if words_list[i] in stop_words: #
            words_list.pop(i)
        elif words_list[i].isdigit():
            words_list.pop(i)
    return words_list
def get_stop_words(path='/home/wzswan/Downloads/tool/DataSet/chinese.txt'):
    file = open(path,'rb').read().decode('utf8').split('\n')
    return set(file)
def rm_char(text):
    text = re.sub('\u3000','',text)
    return text

def svm_classify(train_set,train_tag,test_set,test_tag):

    clf = svm.LinearSVC()
    clf_res = clf.fit(train_set,train_tag)
    train_pred  = clf_res.predict(train_set)
    test_pred = clf_res.predict(test_set)

    train_err_num, train_err_ratio = checkPred(train_tag, train_pred)
    test_err_num, test_err_ratio  = checkPred(test_tag, test_pred)

    print('=== classication is over ===')
    print('traing missing: {e}'.format(e=train_err_ratio))
    print('test missing: {e}'.format(e=test_err_ratio))

    return clf_res


def checkPred(data_tag, data_pred):
    if data_tag.__len__() != data_pred.__len__():
        raise RuntimeError('The length of data tag and data pred should be the same')
    err_count = 0
    for i in range(data_tag.__len__()):
        if data_tag[i]!=data_pred[i]:
            err_count += 1
    err_ratio = err_count / data_tag.__len__()
    return [err_count, err_ratio]

if __name__=='__main__':
    path_doc_root = '/home/wzswan/Downloads/tool/DataSet/THUCNews/THUCNewsTotal' #
    path_tmp = '/home/wzswan/Downloads/tool/DataSet/THUCNews/tmp'  #
    path_dictionary     = os.path.join(path_tmp, 'THUNews.dict')
    path_tmp_tfidf      = os.path.join(path_tmp, 'tfidf_corpus')
    path_tmp_lsi        = os.path.join(path_tmp, 'lsi_corpus')
    path_tmp_lsimodel   = os.path.join(path_tmp, 'lsi_model.pkl')
    path_tmp_predictor  = os.path.join(path_tmp, 'predictor.pkl')
    n = 10  # n

    dictionary = None
    corpus_tfidf = None
    corpus_lsi = None
    lsi_model = None
    predictor = None
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)
    # # ===================================================================
    # # # #
    if not os.path.exists(path_dictionary):
        print('=== did not find dictionary, start mapping to creat dictocary ===')
        dictionary = corpora.Dictionary()
        files = loadFiles(path_doc_root)
        for i,msg in enumerate(files):
            if i%n==0:
                catg    = msg[0]
                file    = msg[1]
                file = convert_doc_to_wordlist(file,cut_all=False)
                dictionary.add_documents([file])
                if int(i/n)%1000==0:
                    print('{t} *** {i} \t docs has been dealed'
                          .format(i=i,t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
        #
        small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < 5 ]
        dictionary.filter_tokens(small_freq_ids)
        dictionary.compactify()
        dictionary.save(path_dictionary)
        print('=== dictionary created ===')
    else:
        print('===if dictionary exists, skip it ===')

    # # ===================================================================
    # # # # tfidf
    if not os.path.exists(path_tmp_tfidf):
        print('=== not find tf ===')
        #
        if not dictionary:  #
            dictionary = corpora.Dictionary.load(path_dictionary)
        os.makedirs(path_tmp_tfidf)
        files = loadFiles(path_doc_root)
        tfidf_model = models.TfidfModel(dictionary=dictionary)
        corpus_tfidf = {}
        for i, msg in enumerate(files):
            if i%n==0:
                catg    = msg[0]
                file    = msg[1]
                word_list = convert_doc_to_wordlist(file,cut_all=False)
                file_bow = dictionary.doc2bow(word_list)
                file_tfidf = tfidf_model[file_bow]
                tmp = corpus_tfidf.get(catg,[])
                tmp.append(file_tfidf)
                if tmp.__len__()==1:
                    corpus_tfidf[catg] = tmp
            if i%10000==0:
                print('{i} files is dealed'.format(i=i))
        #
        catgs = list(corpus_tfidf.keys())
        for catg in catgs:
            corpora.MmCorpus.serialize('{f}{s}{c}.mm'.format(f=path_tmp_tfidf,s=os.sep,c=catg),
                                       corpus_tfidf.get(catg),
                                       id2word = dictionary
                                       )
            print('catg {c} has been transformed into tfidf vector'.format(c=catg))
        print('=== tfidf vector created ===')
    else:
        print('=== ttest the tfidf vector created,skip it ===')

    # # ===================================================================
    # # # #
    if not os.path.exists(path_tmp_lsi):
        print('=== is test the lsi file exists,start to create lsi vector ===')
        if not dictionary:
            dictionary = corpora.Dictionary.load(path_dictionary)
        if not corpus_tfidf: #
            print('--- not get the tfidf documents, start from the disk ---')
            #
            files = os.listdir(path_tmp_tfidf)
            catg_list = []
            for file in files:
                t = file.split('.')[0]
                if t not in catg_list:
                    catg_list.append(t)

            # corpus
            corpus_tfidf = {}
            for catg in catg_list:
                path = '{f}{s}{c}.mm'.format(f=path_tmp_tfidf,s=os.sep,c=catg)
                corpus = corpora.MmCorpus(path)
                corpus_tfidf[catg] = corpus
            print('--- tfidf documents finished, start to chang into lsi vectors ---')

        # lsi model
        os.makedirs(path_tmp_lsi)
        corpus_tfidf_total = []
        catgs = list(corpus_tfidf.keys())
        for catg in catgs:
            tmp = corpus_tfidf.get(catg)
            corpus_tfidf_total += tmp
        lsi_model = models.LsiModel(corpus = corpus_tfidf_total, id2word=dictionary, num_topics=50)
        #
        lsi_file = open(path_tmp_lsimodel,'wb')
        pkl.dump(lsi_model, lsi_file)
        lsi_file.close()
        del corpus_tfidf_total # lsi model
        print('--- lsi model are created---')

        # corpus of lsi,  corpus of tfidf
        corpus_lsi = {}
        for catg in catgs:
            corpu = [lsi_model[doc] for doc in corpus_tfidf.get(catg)]
            corpus_lsi[catg] = corpu
            corpus_tfidf.pop(catg)
            corpora.MmCorpus.serialize('{f}{s}{c}.mm'.format(f=path_tmp_lsi,s=os.sep,c=catg),
                                       corpu,
                                       id2word=dictionary)
        print('=== lsi vectors created ===')
    else:
        print('=== test lsi vectors creted, skip it ===')

    # # ===================================================================
    # # # #
    if not os.path.exists(path_tmp_predictor):
        print('=== not test the judger exists,start the classify processing ===')
        if not corpus_lsi: #
            print('--- not get the lsi document, start to read from disk ---')
            files = os.listdir(path_tmp_lsi)
            catg_list = []
            for file in files:
                t = file.split('.')[0]
                if t not in catg_list:
                    catg_list.append(t)
            # corpus
            corpus_lsi = {}
            for catg in catg_list:
                path = '{f}{s}{c}.mm'.format(f=path_tmp_lsi,s=os.sep,c=catg)
                corpus = corpora.MmCorpus(path)
                corpus_lsi[catg] = corpus
            print('--- lsi document read finished,start to classify ---')

        tag_list = []
        doc_num_list = []
        corpus_lsi_total = []
        catg_list = []
        files = os.listdir(path_tmp_lsi)
        for file in files:
            t = file.split('.')[0]
            if t not in catg_list:
                catg_list.append(t)
        for count,catg in enumerate(catg_list):
            tmp = corpus_lsi[catg]
            tag_list += [count]*tmp.__len__()
            doc_num_list.append(tmp.__len__())
            corpus_lsi_total += tmp
            corpus_lsi.pop(catg)

        #
        data = []
        rows = []
        cols = []
        line_count = 0
        for line in corpus_lsi_total:
            for elem in line:
                rows.append(line_count)
                cols.append(elem[0])
                data.append(elem[1])
            line_count += 1
        lsi_matrix = csr_matrix((data,(rows,cols)), shape=(3, 3)).toarray()
        tag_list = ?
        #
        rarray=np.random.random(size=line_count)
        train_set = []
        train_tag = []
        test_set = []
        test_tag = []
        for i in range(line_count):
            if rarray[i]<0.8:
                train_set.append(lsi_matrix[i,:])
                train_tag.append(tag_list[i])
            else:
                test_set.append(lsi_matrix[i,:])
                test_tag.append(tag_list[i])

        #
        predictor = svm_classify(train_set,train_tag,test_set,test_tag)
        x = open(path_tmp_predictor,'wb')
        pkl.dump(predictor, x)
        x.close()
    else:
        print('=== get classifier created,skip it ===')

    # # ===================================================================
    # # # #
    if not dictionary:
        dictionary = corpora.Dictionary.load(path_dictionary)
    if not lsi_model:
        lsi_file = open(path_tmp_lsimodel,'rb')
        lsi_model = pkl.load(lsi_file)
        lsi_file.close()
    if not predictor:
        x = open(path_tmp_predictor,'rb')
        predictor = pkl.load(x)
        x.close()
    files = os.listdir(path_tmp_lsi)
    catg_list = []
    for file in files:
        t = file.split('.')[0]
        if t not in catg_list:
            catg_list.append(t)
    demo_doc = '/home/wzswan/Downloads/tool/DataSet/newweekdata/130701.txt'
    print("the original text is:")
    print(demo_doc)
    demo_doc = list(jieba.cut(demo_doc,cut_all=False))
    demo_bow = dictionary.doc2bow(demo_doc)
    tfidf_model = models.TfidfModel(dictionary=dictionary)
    demo_tfidf = tfidf_model[demo_bow]
    demo_lsi = lsi_model[demo_tfidf]
    data = []
    cols = []
    rows = []
    for item in demo_lsi:
        data.append(item[1])
        cols.append(item[0])
        rows.append(0)
    demo_matrix = csr_matrix((data,(rows,cols))).toarray()
    x = predictor.predict(demo_matrix)
    print('classication results:{x}'.format(x=catg_list[x[0]]))
