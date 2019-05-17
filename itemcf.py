#-*- coding: utf-8 -*-
'''
Created on 2019-02-28
https://blog.csdn.net/zhangwei15hh/article/details/80502894

@author: Shine
'''

import pandas as pd
import json
import sys
import random
import math
import os
from operator import itemgetter


from collections import defaultdict

random.seed(0)


class ItemBasedCF(object):
    ''' TopN recommendation - Item Based Collaborative Filtering '''

    def __init__(self):
        self.trainset = {}
        self.testset = {}

        self.n_sim_movie = 20   # 近邻数量？
        self.n_rec_movie = 10   # 推荐数量

        self.movie_sim_mat = {} #相似度矩阵？
        self.movie_popular = {} #规模？
        self.movie_count = 0    #什么统计数？
        self.generateDataPath = './'+'generateData'
        if os.path.exists(self.generateDataPath):
            pass
        else:
            os.mkdir(self.generateDataPath)
            print('mkdir %s '%self.generateDataPath)
        print('Similar movie number = %d' % self.n_sim_movie, file=sys.stderr)
        print('Recommended movie number = %d' %
              self.n_rec_movie, file=sys.stderr)

    @staticmethod
    def loadfile(filename): #返回什么？？？？
        ''' load a file, return a generator. '''
#        filename = 'D:\\myGitHub\\KYE_AI\\ecomm\\RecommenderSystem_V1.0\\data\\create_user_item_count.pkl'
        if filename[-4:]=='.txt':   
            print('符合txt')
            with open(filename,'r', encoding='UTF-8') as fp:
                for i, line in enumerate(fp.readlines()):   #enumerate() 返回行索引号和该行内容
#                    print('输出一行',i,line.strip('\r\n'))
                    if i ==0:
                        continue
                    yield line.strip('\r\n')    #yield 节约空间的生成方法，此句作为return结果 
                    if i % 100000 == 0:
                        print ('loading %s(%s)' % (filename, i), file=sys.stderr)   
                fp.close()                
        else:
            fp = open(filename, 'r')
            for i, line in enumerate(fp):   #enumerate() 返回行索引号和该行内容
                yield line.strip('\r\n')    #yield 节约空间的生成方法，此句作为return结果 
                if i % 100000 == 0:
                    print ('loading %s(%s)' % (filename, i), file=sys.stderr)
            fp.close()
        print ('load %s succ' % filename, file=sys.stderr)

    def generate_dataset(self, filename, pivot=0.7):
        ''' load rating data and split it to training set and test set '''
        trainset_len = 0
        testset_len = 0
        
        if filename[-4:]=='.txt':  
            for line in self.loadfile(filename):    #逐行获取yield 传递的结果
                print('file line :',line)
                if trainset_len < 2 :
                    print('filedata, line %s'%trainset_len)
                    print('user, movie, rating')
                    print(line.split(','))
                index, user, movie, rating = line.split(',')
                # split the data by pivot
                if random.random() < pivot:
                    self.trainset.setdefault(user, {})  #dict.setdefault(key, default=None);key -- 查找的键值。default -- 键不存在时，设置的默认键值。
                    self.trainset[user][movie] = int(rating)
                    trainset_len += 1
                    if trainset_len < 2:
                        print('self.trainset[%s]: %s'%(user,self.trainset[user]))
                        print('self.trainset[%s][%s]: %s'%(user,movie,self.trainset[user][movie]))
                        
                else:
                    self.testset.setdefault(user, {})
                    self.testset[user][movie] = int(rating)
                    testset_len += 1            
        else:
            for line in self.loadfile(filename):    #逐行获取yield 传递的结果
                if trainset_len < 2 :
                    print('ratings.dat, line %s'%trainset_len)
                    print('user, movie, rating, _')
                    print(line.split('::'))
                user, movie, rating, _ = line.split('::')
                # split the data by pivot
                if random.random() < pivot:
                    self.trainset.setdefault(user, {})  #dict.setdefault(key, default=None);key -- 查找的键值。default -- 键不存在时，设置的默认键值。
                    self.trainset[user][movie] = int(rating)
                    trainset_len += 1
                    if trainset_len < 2:
                        print('self.trainset[%s]: %s'%(user,self.trainset[user]))
                        print('self.trainset[%s][%s]: %s'%(user,movie,self.trainset[user][movie]))
                        
                else:
                    self.testset.setdefault(user, {})
                    self.testset[user][movie] = int(rating)
                    testset_len += 1
                    
        #   导出训练集，测试集
        with open(self.generateDataPath + "output_trainset.json",'w') as f:
            json.dump(self.trainset,f)
        print('finish dump output_trainset.json')        
        with open(self.generateDataPath + "output_testset.json",'w') as f:
            json.dump(self.testset,f)
        print('finish dump output_testset.json') 

        
        print ('split training set and test set succ', file=sys.stderr)
        print ('train set = %s' % trainset_len, file=sys.stderr)
        print ('test set = %s' % testset_len, file=sys.stderr)

    def calc_movie_sim(self):
        ''' calculate movie similarity matrix '''
        print('counting movies number and popularity...', file=sys.stderr)

        for user, movies in self.trainset.items():  #(Dictionary) items() 函数以列表返回可遍历的(键, 值) 元组数组
            for movie in movies:                    #trainset 为两层字典 ｛user:{movie1:rate1,movie2:rate2,……｝
                # count item popularity
                if movie not in self.movie_popular: #不在movie 集合中
                    self.movie_popular[movie] = 0   #集合中创建该movie 条目
                self.movie_popular[movie] += 1      #movie 计数加1
                              
        print('count movies number and popularity succ', file=sys.stderr)
        
        # 生成movie_popular 表文件
        movie_popular_df = pd.DataFrame(columns = ['movies'],data = list(self.movie_popular.keys()))
        movie_popular_df['count'] = list(self.movie_popular.values())
        movie_popular_df.to_excel('output_movie_popular.xlse')
        print('output movie_popular.xlse')
        # save the total number of movies
        self.movie_count = len(self.movie_popular)
        print('total movie number = %d' % self.movie_count, file=sys.stderr)

        # count co-rated users between items
        itemsim_mat = self.movie_sim_mat
        print('building co-rated users matrix...', file=sys.stderr)

        for user, movies in self.trainset.items():
            for m1 in movies:
                itemsim_mat.setdefault(m1, defaultdict(int))    #defaultdict(int) 效果是key m1 不存在，则｛m1:0｝
                for m2 in movies:
                    if m1 == m2:    #有什么效果呢？？？对角线结果为0？
                        continue
                    itemsim_mat[m1][m2] += 1
       
        print('build co-rated users matrix succ', file=sys.stderr)
        # itemsim_mat[m1][m2] = value 生成邻接链表
        # calculate similarity matrix
        print('calculating movie similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000
        for m1, related_movies in itemsim_mat.items():
            for m2, count in related_movies.items():
                #   更新 itemsim_mat，m1*m2/(m1*m2)^(1/2)为余弦相似度，可以改进为其他相似度计算法验证效果：皮尔逊相关系数，斯皮尔曼等级数，Jaccard公式
                itemsim_mat[m1][m2] = count / math.sqrt(
                    self.movie_popular[m1] * self.movie_popular[m2])
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:   # 每两百万次要提醒？？
                    print('calculating movie similarity factor(%d)' %
                          simfactor_count, file=sys.stderr)
        #导出json
        with open("output_itemsim_mat.json",'w') as f:
            json.dump(itemsim_mat,f)
        print('finish dump output_itemsim_mat.json')

        #   更新后的itemsim_mat
        print('calculate movie similarity matrix(similarity factor) succ',
              file=sys.stderr)
        #   总计算量
        print('Total similarity factor number = %d' %
              simfactor_count, file=sys.stderr)

    def recommend(self, user):
        ''' Find K similar movies and recommend N movies. '''
        K = self.n_sim_movie    #邻近数
        N = self.n_rec_movie    #推荐数
        rank = {}
        watched_movies = self.trainset[user]    #watched_movies 是第二层dict ，user 看过的多部电影及对应评分

        for movie, rating in watched_movies.items():
            #   此处存了双向，但实际只需要一半的空间即可，mat[m1m2]=mat[m2m1]
            #   sorted(可迭代对象,key,reverse)key为按哪个排序，reverse =true 从大到小排,itemgetter 取兑现的维度/域，itemgetter（1） 取对象第二维
            for related_movie, similarity_factor in sorted(self.movie_sim_mat[movie].items(),
                                                           key=itemgetter(1), reverse=True)[:K]:    #取K近邻
                if related_movie in watched_movies:
                    continue
                #相关电影user 没评价过才推荐
                rank.setdefault(related_movie, 0)
                rank[related_movie] += similarity_factor * rating   #新电影与原始电影相似度*原始电影评分
        # return the N best movies
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

    def evaluate(self):
        ''' print evaluation result: precision, recall, coverage and popularity '''
        print('Evaluation start...', file=sys.stderr)

        N = self.n_rec_movie
        #  varables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_movies = set()
        # varables for popularity
        popular_sum = 0

        for i, user in enumerate(self.trainset):    #enumerate(dict) 返回 index 和 key
            if i % 500 == 0:
                print ('recommended for %d users' % i, file=sys.stderr)     #每500个用户推荐提醒一次
            test_movies = self.testset.get(user, {})    #user 不存在时返回默认{}//testset 会存在 trainset 的user 吗？？
            rec_movies = self.recommend(user)   #推荐的N个结果
            if i<2:
                print('test_movies:',test_movies)
            for movie, _ in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)   #重复推荐同一个的时候呢？
                # 计算平均流行度，使用log是为了使均值比较稳定，moviepopularity里面存储的是每个电影被正反馈操作的次数。
                popular_sum += math.log(1 + self.movie_popular[movie])  #推荐电影频繁度log 后累计
            rec_count += N      #每个人加一个N ,直接 len(trainset)*N 即可
            test_count += len(test_movies)  #测试用户喜爱电影总数

        #
        precision = hit / (1.0 * rec_count)                         #准确率 
        recall = hit / (1.0 * test_count)                           #召回率 需要考虑用户喜欢列表过长时对召回率计算效果不佳的影响
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)   #覆盖率
        popularity = popular_sum / (1.0 * rec_count)                #？？？

        print ('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %
               (precision, recall, coverage, popularity), file=sys.stderr)


if __name__ == '__main__':
#    ratingfile = os.path.join('ml-1m', 'ratings.dat')

#    'D:\myGitHub\KYE_AI\ecomm\RecommenderSystem_V1.0\data\create_user_item_count.pkl'
    ratingfile = os.path.join('data', 'create_user_item_times.txt')
    
    itemcf = ItemBasedCF()
    itemcf.generate_dataset(ratingfile)
    itemcf.calc_movie_sim()
    itemcf.evaluate()
