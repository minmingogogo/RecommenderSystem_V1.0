# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 14:58:31 2019
https://blog.csdn.net/qq_33592583/article/details/79417189
@author: shine
"""


import pandas as pd
import numpy as np
import json
import sys
import random
import math
import os
from operator import itemgetter
import copy
from config.common import logger
from collections import defaultdict
from sklearn import cross_validation
import time

random.seed(0)


class ItemBasedCF(object):
    ''' TopN recommendation - Item Based Collaborative Filtering '''

    def __init__(self,n_sim_item = 20,n_rec_item = 10 ):


        self.n_sim_item = n_sim_item   # 近邻数量？
        self.n_rec_item = n_rec_item   # 推荐数量


        self.generateDataPath = './'+'generateData'
        if os.path.exists(self.generateDataPath):
            pass
        else:
            os.mkdir(self.generateDataPath)
            print('mkdir %s '%self.generateDataPath)        
        print('Similar item number = %d' % self.n_sim_item, file=sys.stderr)
        print('Recommended item number = %d' %self.n_rec_item, file=sys.stderr)
        logger.info('Similar item number = %d' % self.n_sim_item)
        logger.info('Recommended item number = %d' %self.n_rec_item)


    def loadfile(self,filename,itemfile): #返回什么？？？？
        ''' load a file, return a generator. '''
#        filename = 'D:\\myGitHub\\KYE_AI\\ecomm\\RecommenderSystem_V1.0\\data\\create_Dataframe_user_item_times_Merge.xlsx'
        if filename[-4:] == '.pkl':
            data = pd.read_pickle(filename)
        elif filename[-5:] == '.xlsx':
            data = pd.read_excel(filename)
        print ('load %s succ' % filename, file=sys.stderr)
        #   设置数据类型
        try:
            data = data.astype({'user':'str','item':'str'})    
        except :
            data = data.rename(columns = {'用户ID':'user','商品ID':'item'})
            data = data.astype({'user':'str','item':'str'}) 
        self.itemmap = pd.read_excel(itemfile)
        self.itemmap = self.itemmap.rename(columns = {'商品ID':'item'})
        self.itemmap = self.itemmap.astype({'item':'str'}) 
        return data

    def createTrees_User_Item_times(self,dataframe,lev1,lev2,val):
        num = len(dataframe)
#        dataframe = dataframe.astype({lev1:'str',lev2:'str'})
        print('待处理数：%s'%num)
        user_item_Dict = {}
        s_t = time.clock()
        for ind in range(len(dataframe)):
            user = dataframe[lev1][ind]
            item = dataframe[lev2][ind]
            rating = dataframe[val][ind]
            try:
                user_item_Dict[user]
            except:            
                user_item_Dict.setdefault(user,{})
            try:
                user_item_Dict[user][item]
            except:            
                user_item_Dict[user].setdefault(item,1)
            else:
                user_item_Dict[user][item] += int(rating)
            if ind % 10000 == 0 or ind == num-1:            
                print('已处理 %s 个 已耗时 %s 分钟'%(str(ind),str((time.clock()-s_t)/60)))
        return user_item_Dict


    def generate_dataset(self, filename,itemfile, testSize = 0.3,lev1_ActiveScore=3,lev2_ActiveScore = 2, lev3_ActiveScore= 1):
        ''' load rating data and split it to training set and test set '''
        self.trainset = {}
        self.testset = {}
        trainset_len = 0
        testset_len = 0
        #   权重调整评分
        data = self.loadfile(filename,itemfile)
        self.rowdata = copy.deepcopy(data)
        #   调整乘数比例
        data['times_Scale'] = np.where(np.array(data['active'])=='订单',lev1_ActiveScore,np.where(np.array(data['active'])=='浏览',lev3_ActiveScore,lev2_ActiveScore))
        data['times_Modify'] = data['times']*data['times_Scale']        
        #   数据集分离
        try :
            x_train,x_test,y_train,y_test = cross_validation.train_test_split(data[['item', 'times_Modify', 'user']],data[['active']], test_size = testSize, random_state = 0)
        except Exception as e:
            logger.error('train_test_split error : %s'%e)
            print('[X]train_test_split error')
        else:
            trainset_len = len(x_train)
            testset_len = len(x_test)
        self.trainDf = x_train.reset_index(drop = True)
        self.testDf = x_test.reset_index(drop = True)           
            
        self.trainset = self.createTrees_User_Item_times(self.trainDf,'user','item','times_Modify')
        self.testset = self.createTrees_User_Item_times(self.testDf,'user','item','times_Modify')

        #   导出训练集，测试集
        with open(self.generateDataPath +'/'+ "output_trainset.json",'w') as f:
            json.dump(self.trainset,f)
        print('finish dump output_trainset.json')        
        with open(self.generateDataPath + '/'+"output_testset.json",'w') as f:
            json.dump(self.testset,f)
        print('finish dump output_testset.json') 
        print ('split training set and test set succ', file=sys.stderr)
        print ('train set = %s' % trainset_len, file=sys.stderr)
        print ('test set = %s' % testset_len, file=sys.stderr)
    
    def generate_itemPopular(self):

        self.item_popular = {} #规模？
        self.item_count = 0    #什么统计数？
        
        #   构造item_popular 集合 value 为 次数，用x_train即可
        #   item 用户数
        item_user_count_df = self.trainDf[['item','user']].sort_values(by = 'item').drop_duplicates()
        item_user_count = pd.DataFrame(item_user_count_df['item'].value_counts()).reset_index(drop = False).rename(columns = {'index':'item','item':'user_count'})
        #   item 总分数
        item_times_count = pd.DataFrame(pd.pivot_table(self.trainDf,index = 'item',values = 'times_Modify',aggfunc = np.sum)).reset_index(drop = False).sort_values(by = 'item')
        #   item 平均分 皮尔逊相关系数用
        item_statistics = pd.merge(item_user_count,item_times_count,on = 'item',how = 'left')
        item_statistics['times_avg'] = item_statistics['times_Modify']/item_statistics['user_count']
        item_statistics = item_statistics.astype({'times_Modify':'str','times_avg':'str'})
        item_popular_df = item_statistics
        self.item_statistics = item_statistics
        for ind in range(len(item_popular_df)):
            item = item_popular_df['item'][ind]
            popular = item_popular_df['times_Modify'][ind]
            avg = item_popular_df['times_avg'][ind]
            if item not in self.item_popular:
                self.item_popular[item] = {'sum':popular,'avg':avg}
                
        #   按流行度排序
        item_popular_df = item_popular_df.sort_values(by = 'times_Modify',axis = 0,ascending = False)
        print('count items number and popularity succ', file=sys.stderr)
        
        # 生成item_popular 表文件
        item_popular_df.to_excel('output_item_popular.xlsx')
        print('output item_popular.xlsx')
        #导出json int 转 str
        with open(self.generateDataPath + '/'+ "output_item_popular.json",'w') as f:
            json.dump(self.item_popular,f)
        print('finish dump output_item_popular.json')
        
        # save the total number of items
        self.item_count = len(item_popular_df)
        print('total item number = %d' % self.item_count, file=sys.stderr)

    def calc_item_sim_cos(self):
        ''' calculate item similarity matrix '''
        self.item_sim_mat = {} #相似度矩阵？
        
        #   余弦相似度计算 :user1 : item1 :5, item2:3 ,N(item1 * item2) = min(5,3):
            #   解释为： 一个用户，浏览item1 5次，item2 3次，则item1 item2 交集为3次
        print('counting items number and popularity...', file=sys.stderr)
        # count co-rated users between items
        itemsim_mat = self.item_sim_mat
        print('building co-rated users matrix...', file=sys.stderr)

        for user, items in self.trainset.items():
            for m1 in items:
                itemsim_mat.setdefault(m1, defaultdict(int))    #defaultdict(int) 效果是key m1 不存在，则｛m1:0｝
                for m2 in items:
                    if m1 == m2:    #邻接链表存储，矩阵对角线结果为1
                        continue
                    itemsim_mat[m1][m2] += min(self.trainset[user][m1],self.trainset[user][m2])     #集合分数为乘以权重以后的，此处取m1 m2 的交集
        
        print('build co-rated users matrix succ', file=sys.stderr)
        # itemsim_mat[m1][m2] = value 生成邻接链表
        # calculate similarity matrix
        print('calculating item similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000
        for m1, related_items in itemsim_mat.items():
            for m2, count in related_items.items():
                #   更新 itemsim_mat，m1*m2/(m1*m2)^(1/2)为余弦相似度，可以改进为其他相似度计算法验证效果：皮尔逊相关系数，斯皮尔曼等级数，Jaccard公式

                if int(self.item_popular[m1]['sum']) * int(self.item_popular[m2]['sum']) == 0:
                    print('m1 :%s ,m2 : %s'%(m1,m2))
                    print('item_popular[m1][sum]:%s'%self.item_popular[m1]['sum'])
                    logger.error('item_popular[m1][sum]:%s'%self.item_popular[m1]['sum'])
                itemsim_mat[m1][m2] = count / math.sqrt(
                    int(self.item_popular[m1]['sum']) * int(self.item_popular[m2]['sum']))
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:   # 每两百万次要提醒？？
                    print('calculating item similarity factor(%d)' %
                          simfactor_count, file=sys.stderr)
        #导出json
        with open(self.generateDataPath + '/'+ "output_itemsim_cos_mat.json",'w') as f:
            json.dump(itemsim_mat,f)
        print('finish dump output_itemsim_mat.json')
        #   更新后的itemsim_mat
        print('calculate item similarity matrix(similarity factor) succ',file=sys.stderr)
        #   总计算量
        print('Total similarity factor number = %d' %simfactor_count, file=sys.stderr)


    def calc_item_sim_personCor(self):
        ''' calculate item similarity matrix '''
        self.item_sim_mat = {} #相似度矩阵？
        #   余弦相似度计算 :user1 : item1 :5, item2:3 ,N(item1 * item2) = min(5,3):
            #   解释为： 一个用户，浏览item1 5次，item2 3次，则item1 item2 交集为3次
        print('counting items number and popularity...', file=sys.stderr)
        
        #   create item-item Mat
        itemNum = len(self.item_popular)
        item_index = pd.DataFrame(list(self.item_popular.keys())).reset_index(drop = False).rename(columns = {0:'item'})     # 矩阵索引
        
#        item_index = pd.DataFrame(list(item_statistics['item'])).reset_index(drop = False).rename(columns = {0:'item'})     # 矩阵索引
        item_index_Dict = {}
        for ind in range(len(item_index)):
            item_index_Dict[item_index['item'][ind]] = item_index['index'][ind]
                                 
#        sim_mat = np.zeros((itemNum,itemNum))
        sim_mat = np.diag([1]*len(self.item_popular)) 
        print('create sim_mat:')
        print(sim_mat)
        #   index 与 itemID 对应表
        # count co-rated users between items
        print('building co-rated users matrix...', file=sys.stderr)
        itemsim_mat = {}
        for user, items in self.trainset.items():
            for m1 in items:
                if itemsim_mat.get(m1) == None:
                    itemsim_mat.setdefault(m1, {})                    
                for m2 in items:
                    if m1 == m2:    #邻接链表存储，矩阵对角线结果为1
                        continue
                    sim_mat[item_index_Dict[m1]][item_index_Dict[m2]] += min(self.trainset[user][m1],self.trainset[user][m2])     #集合分数为乘以权重以后的，此处取m1 m2 的交集
                    if itemsim_mat[m1].get(m2) != None:
                        continue
                    else:
                        itemsim_mat[m1][m2] = 1 #该记录用于后续调整存出的矩阵为邻接链表

        print('build co-rated users matrix succ', file=sys.stderr)
        # itemsim_mat[m1][m2] = value 生成邻接链表
        # calculate similarity matrix
        print('calculating item Person correlation similarity matrix...', file=sys.stderr)
        sim_mat = np.corrcoef(sim_mat)
        print('皮尔逊相关系数矩阵corrcoef:')
        print(sim_mat)
        simfactor_count = 0
        self.item_sim_mat = copy.deepcopy(itemsim_mat)
        for m1 in list(itemsim_mat.keys()):
            for m2 in list(itemsim_mat[m1].keys()):                    
                self.item_sim_mat[m1][m2] = sim_mat[item_index_Dict[m1]][item_index_Dict[m2]]
                if simfactor_count<2:
                    print('item_index_Dict[m1] : %s ,item_index_Dict[m2] :%s'%(item_index_Dict[m1],item_index_Dict[m2]))
                simfactor_count +=1
        
        #导出json
        with open(self.generateDataPath + '/'+ "output_itemsim_personCor_mat.json",'w') as f:
            json.dump(itemsim_mat,f)
        print('finish dump output_itemsim_personCor_mat.json')
        #   更新后的itemsim_mat
        print('calculate item similarity matrix(similarity factor) succ',file=sys.stderr)
        #   总计算量
        print('Total similarity factor number = %d' %simfactor_count, file=sys.stderr)


    def recommend(self, user):
        ''' Find K similar items and recommend N items. '''
        K = self.n_sim_item    #邻近数
        N = self.n_rec_item    #推荐数
        rank = {}
        watched_items = self.trainset[user]    #watched_items 是第二层dict ，user 看过的多部电影及对应评分
        for item, rating in watched_items.items():
            #   此处存了双向，但实际只需要一半的空间即可，mat[m1m2]=mat[m2m1]
            #   sorted(可迭代对象,key,reverse)key为按哪个排序，reverse =true 从大到小排,itemgetter 取兑现的维度/域，itemgetter（1） 取对象第二维
            for related_item, similarity_factor in sorted(self.item_sim_mat[item].items(),
                                                           key=itemgetter(1), reverse=True)[:K]:    #取K近邻
                if related_item in watched_items:
                    continue
                #相关电影user 没评价过才推荐
                rank.setdefault(related_item, 0)
                rank[related_item] += similarity_factor * rating   #新电影与原始电影相似度*原始电影评分
        # return the N best items
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

    def recommend_includeHaveBuy(self, user):
        ''' Find K similar items and recommend N items. '''
        K = self.n_sim_item    #邻近数
        N = self.n_rec_item    #推荐数
        rank = {}
        watched_items = self.trainset[user]    #watched_items 是第二层dict ，user 看过的多部电影及对应评分
        for item, rating in watched_items.items():
            #   此处存了双向，但实际只需要一半的空间即可，mat[m1m2]=mat[m2m1]
            #   sorted(可迭代对象,key,reverse)key为按哪个排序，reverse =true 从大到小排,itemgetter 取兑现的维度/域，itemgetter（1） 取对象第二维
            for related_item, similarity_factor in sorted(self.item_sim_mat[item].items(),
                                                           key=itemgetter(1), reverse=True)[:K]:    #取K近邻
                #相关电影user 没评价过才推荐
#                if related_item in watched_items:
#                    continue
                rank.setdefault(related_item, 0)
                rank[related_item] += similarity_factor * rating   #新电影与原始电影相似度*原始电影评分
        # return the N best items
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]


    def evaluate(self):
        ''' print evaluation result: precision, recall, coverage and popularity '''
        print('Evaluation start...', file=sys.stderr)

        N = self.n_rec_item
        #  varables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_items = set()
        # varables for popularity
        popular_sum = 0
        
        testResult = pd.DataFrame()
        for i, user in enumerate(self.trainset):    #enumerate(dict) 返回 index 和 key
            trainsetItemList = []
            testsetItemList = []
            recommendList = []  
            hitItemList = []                  
            if self.trainset.get(user,{}) == {}:
                print('%s data error'%user)
            else:
                trainsetItemList = list(self.trainset[user].keys())
            if i % 500 == 0:
                print ('recommended for %d users' % i, file=sys.stderr)     #每500个用户推荐提醒一次
            test_items = self.testset.get(user, {})    #user 不存在时返回默认{}//testset 会存在 trainset 的user 吗？？
            if test_items != {}: #无这个user
                testsetItemList = list(self.testset[user].keys())
            rec_items = self.recommend(user)   #推荐的N个结果
            if i<2:
                print('test_items:',test_items)
            for item, _ in rec_items:
                recommendList.append(item)
                if item in test_items:
                    hit += 1
                    hitItemList.append(item)
                all_rec_items.add(item)   #重复推荐同一个的时候呢？
                # 计算平均流行度，使用log是为了使均值比较稳定，itempopularity里面存储的是每个电影被正反馈操作的次数。
                popular_sum += math.log(1 + int(self.item_popular[item]['sum']))  #推荐电影频繁度log 后累计
            rec_count += N      #每个人加一个N ,直接 len(trainset)*N 即可
            test_count += len(test_items)  #测试用户喜爱电影总数
            subdf = pd.DataFrame({'user':[user],'hitnum':[len(hitItemList)],'hit':[hitItemList],'recomm':[recommendList],'trainItem':[trainsetItemList],'testItem':[testsetItemList]})
            testResult = pd.concat([testResult,subdf])
        testResult.reset_index(drop = True,inplace = True)
        #
        precision = hit / (1.0 * rec_count)                         #准确率 
        recall = hit / (1.0 * test_count)                           #召回率 需要考虑用户喜欢列表过长时对召回率计算效果不佳的影响
        coverage = len(all_rec_items) / (1.0 * self.item_count)   #覆盖率
        popularity = popular_sum / (1.0 * rec_count)                #流行度

        print ('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %
               (precision, recall, coverage, popularity), file=sys.stderr)
        #   导出
        testResult.to_excel(self.generateDataPath + '/'+"testResult.xlsx")
        return 'finish'

    def analyse(self):
        ''' print evaluation result: precision, recall, coverage and popularity '''
        print('Analyse start...', file=sys.stderr)

        N = self.n_rec_item

        #   统计数据 ：test 中有但train 中没有的用户，test 中有train 中没有的商品
        #   统计数据 ：train 中有但test 中没有的用户，train 中有test 中没有的商品
        train_userSet = set(self.trainset.keys())
        test_userSet = set(self.testset.keys())
        train_itemSet = set(self.trainDf['item'])
        test_itemSet = set(self.testDf['item'])
        
        user_diff_train = train_userSet.difference(test_userSet)
        user_diff_test = test_userSet.difference(train_userSet)
        item_diff_train = train_itemSet.difference(test_itemSet)
        item_diff_test = test_itemSet.difference(train_itemSet)
        
        user_interset = train_userSet.intersection(test_userSet)
        user_unionset = train_userSet.union(test_userSet)
        item_interset = train_itemSet.intersection(test_itemSet)
        item_unionset = train_itemSet.union(test_itemSet)

        print('train user number : %s ,test user number : %s , intersetion number : %s '%(len(train_userSet),len(test_userSet),len(user_interset)))
        print('train user ratio : %s ,test user ratio : %s '%(len(train_userSet)/len(user_unionset),len(test_userSet)/len(user_unionset)))
        print('train user diff from test number : %s ,ratio : %s'%(len(user_diff_train),len(user_diff_train)/len(test_userSet)))
        print('test user diff from train number : %s ,ratio : %s'%(len(user_diff_test),len(user_diff_test)/len(train_userSet)))

        print('train item number : %s ,test item number : %s , intersetion number : %s '%(len(train_itemSet),len(test_itemSet),len(item_interset)))
        print('train item ratio : %s ,test item ratio : %s '%(len(train_itemSet)/len(item_unionset),len(test_itemSet)/len(item_unionset)))
        print('train item diff from test number : %s ,ratio : %s'%(len(item_diff_train),len(item_diff_train)/len(test_itemSet)))
        print('test item diff from train number : %s ,ratio : %s'%(len(item_diff_test),len(item_diff_test)/len(train_itemSet)))

        
        logger.info('train user number : %s ,test user number : %s , intersetion number : %s '%(len(train_userSet),len(test_userSet),len(user_interset)))
        logger.info('train user ratio : %s ,test user ratio : %s '%(len(train_userSet)/len(user_unionset),len(test_userSet)/len(user_unionset)))
        logger.info('train user diff from test number : %s ,ratio : %s'%(len(user_diff_train),len(user_diff_train)/len(test_userSet)))
        logger.info('test user diff from train number : %s ,ratio : %s'%(len(user_diff_test),len(user_diff_test)/len(train_userSet)))

        logger.info('train item number : %s ,test item number : %s , intersetion number : %s '%(len(train_itemSet),len(test_itemSet),len(item_interset)))
        logger.info('train item ratio : %s ,test item ratio : %s '%(len(train_itemSet)/len(item_unionset),len(test_itemSet)/len(item_unionset)))
        logger.info('train item diff from test number : %s ,ratio : %s'%(len(item_diff_train),len(item_diff_train)/len(test_itemSet)))
        logger.info('test item diff from train number : %s ,ratio : %s'%(len(item_diff_test),len(item_diff_test)/len(train_itemSet)))

        #   测试集合： 1训练集没有的用户，2训练集中有的用户 两类表现
        
#        if len(user_diff_test) > 0:
#            print('待处理用户数：%s'%len(user_diff_test))
#            
#            testResult = pd.DataFrame()
#            testResult_Dict = {}
#            #  varables for precision and recall
#            hit = 0
#            rec_count = 0
#            test_count = 0
#            # varables for coverage
#            all_rec_items = set()
#            # varables for popularity
#            popular_sum = 0        
#            #   训练集中没有的用户
#            for i in range(len(user_diff_test)):
#                user = list(user_diff_test)[i]
#                trainsetItemList = []
#                testsetItemList = []
#                recommendList = []
#                simScoreList = []
#                hitItemList = []              
#                test_items = self.testset.get(user, {})    #user 不存在时返回默认{}//testset 会存在 trainset 的user 吗？？
#                testResult_Dict[user] = {'testSet':list(self.testset[user].keys())}    
#                if test_items != {}: #无这个user
#                    testsetItemList = list(self.testset[user].keys())
#                rec_items = self.recommend(user)   #推荐的N个结果
#                for item, score in rec_items:
#                    recommendList.append(item)
#                    simScoreList.append(score)
#                    if item in test_items:
#                        hit += 1
#                        hitItemList.append(item)
#                    all_rec_items.add(item)   #重复推荐同一个的时候呢？
#                    popular_sum += math.log(1 +  int(self.item_popular[item]['sum']))  #推荐电影频繁度log 后累计
#                testResult_Dict[user]['recommendSet'] = recommendList
#                testResult_Dict[user]['hitItemSet'] = hitItemList
#                testResult_Dict[user]['hitnum'] = len(hitItemList)
#                testResult_Dict[user]['simScoreSet'] = simScoreList
#                rec_count += N      #每个人加一个N ,直接 len(trainset)*N 即可
#                test_count += len(test_items)  #测试用户喜爱电影总数
#                subdf = pd.DataFrame({'user':[user],'hitnum':[len(hitItemList)],'hit':[hitItemList],'recomm':[recommendList],'trainItem':[trainsetItemList],'testItem':[testsetItemList],'simScore':[simScoreList]})
#                testResult = pd.concat([testResult,subdf])
#            testResult.reset_index(drop = True,inplace = True)
#            #
#            precision = hit / (1.0 * rec_count)                         #准确率 
#            recall = hit / (1.0 * test_count)                           #召回率 需要考虑用户喜欢列表过长时对召回率计算效果不佳的影响
#            coverage = len(all_rec_items) / (1.0 * self.item_count)   #覆盖率
#            popularity = popular_sum / (1.0 * rec_count)                #流行度    
#            print ('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %
#                   (precision, recall, coverage, popularity), file=sys.stderr) 
#            
#            logger.info('[训练集中没有的] precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %(precision, recall, coverage, popularity))
#        #   导出
#            testResult.to_excel(self.generateDataPath + '/'+"testResult_notrain.xlsx")
#            #导出json
#            with open(self.generateDataPath + '/'+ "output_item_popular_notrain.json",'w') as f:
#                json.dump(self.item_popular,f)
#            print('finish dump output_item_popular.json')

        #   训练集中有的用户
        if len(user_interset) > 0:
            print('待处理用户数：%s'%len(user_interset))
            testResult = pd.DataFrame()
            testResult_Dict = {}
            #  varables for precision and recall
            hit = 0
            rec_count = 0
            test_count = 0
            # varables for coverage
            all_rec_items = set()
            # varables for popularity
            popular_sum = 0        
            #   训练集中没有的用户
            for i in range(len(user_interset)):
                user = list(user_interset)[i]
                trainsetItemList = []
                testsetItemList = []
                recommendList = []
                simScoreList = []
                hitItemList = []              
                test_items = self.testset.get(user, {})    #user 不存在时返回默认{}//testset 会存在 trainset 的user 吗？？
                if test_items != {}: #无这个user
                    testsetItemList = list(self.testset[user].keys())
                testResult_Dict[user] = {'testSet':list(self.testset[user].keys())}
                trainsetItemList = list(self.trainset[user].keys())
                testResult_Dict[user] = {'trainSet':trainsetItemList}                    
                    
                rec_items = self.recommend(user)   #推荐的N个结果
                for item, score in rec_items:
                    recommendList.append(item)
                    simScoreList.append(score)
                    if item in test_items:
                        hit += 1
                        hitItemList.append(item)
                    all_rec_items.add(item)   #重复推荐同一个的时候呢？
                    popular_sum += math.log(1 + int(self.item_popular[item]['sum']))  #推荐电影频繁度log 后累计
                testResult_Dict[user]['recommendSet'] = recommendList
                testResult_Dict[user]['hitItemSet'] = hitItemList
                testResult_Dict[user]['hitnum'] = len(hitItemList)
                testResult_Dict[user]['simScoreSet'] = simScoreList
                rec_count += N      #每个人加一个N ,直接 len(trainset)*N 即可
                test_count += len(test_items)  #测试用户喜爱电影总数
                subdf = pd.DataFrame({'user':[user],'hitnum':[len(hitItemList)],'hit':[hitItemList],'recomm':[recommendList],'trainItem':[trainsetItemList],'testItem':[testsetItemList],'simScore':[simScoreList]})
                testResult = pd.concat([testResult,subdf])
            testResult.reset_index(drop = True,inplace = True)
            #
            precision = hit / (1.0 * rec_count)                         #准确率 
            recall = hit / (1.0 * test_count)                           #召回率 需要考虑用户喜欢列表过长时对召回率计算效果不佳的影响
            coverage = len(all_rec_items) / (1.0 * self.item_count)     #覆盖率
            popularity = popular_sum / (1.0 * rec_count)                #流行度    
            print ('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %
                   (precision, recall, coverage, popularity), file=sys.stderr) 
            logger.info('[训练集中有的] precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %(precision, recall, coverage, popularity))
        #   导出
            testResult.to_excel(self.generateDataPath + '/'+"testResult_withtrain.xlsx")
            #导出json
            with open(self.generateDataPath + '/'+ "output_item_testResult_withtrain.json",'w') as f:
                json.dump(self.item_popular,f)
            print('finish dump output_item_popular.json')
        #--------------------------------------------------------

        return precision,recall,coverage,popularity
        
    def recommendInfo(self,user,item = None):
        print('用户%s 推荐商品情况'%user)
        print('1.用户历史行为：')
        temp = self.rowdata[self.rowdata['user']==user]
        mergeTemp = pd.merge(temp,self.itemmap,on='item',how = 'left')

        print(mergeTemp[['active','一级分类','二级分类','标题']])
        print('2.推荐商品（排除已有行为的商品）：')
        rec_items = self.recommend(user)   #推荐的N个结果
        recommendList = []
        simScoreList = []
        for item, score in rec_items:
            recommendList.append(item)
            simScoreList.append(score)        
        rec_Df = pd.DataFrame({'item':recommendList,'score':simScoreList})
        merge_rec_1 = pd.merge(rec_Df,self.itemmap,on='item',how = 'left')
        print(merge_rec_1[['一级分类','二级分类','score','标题']])
        print('3.推荐商品（包含已有行为的商品）：')
        rec_items = self.recommend_includeHaveBuy(user)   #推荐的N个结果
        recommendList = []
        simScoreList = []
        for item, score in rec_items:
            recommendList.append(item)
            simScoreList.append(score)        
        rec_Df = pd.DataFrame({'item':recommendList,'score':simScoreList})
        merge_rec_2 = pd.merge(rec_Df,self.itemmap,on='item',how = 'left')
        print(merge_rec_2[['一级分类','二级分类','score','标题']])
        
#        print('4.当前商品最相似的Top 5 ：')
#    #   当前状态商品top 5
        item_TopSim = []
#        if item != None:
#            print(self.itemmap.iloc[[list(self.itemmap['item']).index(item)]])
#            temp = pd.DataFrame({'item':list(self.item_sim_mat[item].keys()),'score':list(self.item_sim_mat[item].values())}).sort_values(by = 'score',ascending = False)
#            item_TopSim = pd.merge(temp.head(10),self.itemmap,on='item',how = 'left')
#            print(item_TopSim)
                    
        return mergeTemp,merge_rec_1,merge_rec_2,item_TopSim
            
if __name__ == '__main__':
#    ratingfile = os.path.join('ml-1m', 'ratings.dat')

#    ratingfile = 'D:\myGitHub\KYE_AI\ecomm\RecommenderSystem_V1.0\data\create_Dataframe_user_item_times_Merge.xlsx'
#    itemfile = 'D:\myGitHub\KYE_AI\ecomm\RecommenderSystem_V1.0\data\商品标题及类目表.xlsx'
    ratingfile = os.path.join('data', 'create_Dataframe_user_item_times_Merge.xlsx')
    itemfile = os.path.join('data', '商品标题及类目表.xlsx')
#==============================================================================
# 单用户 单商品 验证
#==============================================================================

    num = 20
    itemcf = ItemBasedCF(n_sim_item = num)
    itemcf.generate_dataset(ratingfile,itemfile,testSize = 0)
    itemcf.generate_itemPopular()
    itemcf.calc_item_sim_cos()
    
#    user = '1006458808717869952'
    user = itemcf.rowdata['user'][855]
    mergeTemp,merge_rec_1,merge_rec_2,item_TopSim = itemcf.recommendInfo(user)     

    item = list(itemcf.item_popular.keys())[5]
    mergeTemp,merge_rec_1,merge_rec_2,item_TopSim = itemcf.recommendInfo(user,item = item)     

    #   当前状态商品top 5
    item = list(itemcf.item_popular.keys())[random.randint(0,itemcf.item_count-1)]
    print(itemcf.itemmap.iloc[[list(itemcf.itemmap['item']).index(item)]])
    temp = pd.DataFrame({'item':list(itemcf.item_sim_mat[item].keys()),'score':list(itemcf.item_sim_mat[item].values())}).sort_values(by = 'score',ascending = False)
    item_TopSim = pd.merge(temp.head(),itemcf.itemmap,on='item',how = 'left')
    print(item_TopSim)

#==============================================================================
#  参数测试
#==============================================================================

    #参数1：近邻数 n_sim_item，
    #参数2：行为权重 lev1_ActiveScore=3,lev2_ActiveScore = 2, lev3_ActiveScore= 1
    #参数3 ：testSize = 0.3
    n_sim_itemL = [5,10,15,20,30,40]
    lev_ActiveScoreL = [[1,1,1],
                        [3,2,1],
                        [10,5,1],
                        [100,10,1]]
    
    testSizeL = [0.3]
    
    analyseDf = pd.DataFrame()
    for ind_testSize in range(len(testSizeL)):
        testSize = testSizeL[ind_testSize]
        for ind_n_sim_itemL in range(len(n_sim_itemL)):
            num = n_sim_itemL[ind_n_sim_itemL]
            for ind_lev in range(len(lev_ActiveScoreL)):
                lev1_s,lev2_s,lev3_s = lev_ActiveScoreL[ind_lev]
                
                itemcf = ItemBasedCF(n_sim_item = num)
                itemcf.generate_dataset(ratingfile,itemfile,testSize = testSize,lev1_ActiveScore = lev1_s,lev2_ActiveScore = lev2_s, lev3_ActiveScore= lev3_s)
                itemcf.generate_itemPopular() #不调整优先级分数 不需要更新itempopular
                itemcf.calc_item_sim_cos()
                precision,recall,coverage,popularity = itemcf.analyse()                
                
                subdf = pd.DataFrame({'抽样比例':[testSize],'近邻数':[num],'一级权重':[lev1_s],'二级权重':[lev2_s],'三级权重':[lev3_s],'precision':[precision],'recall':[recall],'coverage':[coverage],'popularity':[popularity]})
                analyseDf = pd.concat([analyseDf,subdf])
    analyseDf.reset_index(drop = True,inplace =True)
    analyseDf.to_excel('D:\myGitHub\KYE_AI\ecomm\RecommenderSystem_V1.0\generateData\交叉测试结果.xlsx')
                
                
#==============================================================================
#  测试 user item 推荐结果 
#==============================================================================
    filepath = 'D:\myGitHub\KYE_AI\ecomm\RecommenderSystem_V1.0\generateData'
    filename = 'D:\myGitHub\KYE_AI\ecomm\RecommenderSystem_V1.0\data' + '/'+'数据举例.xlsx' 
    sheetname = '用户数据'    
    userDF = pd.read_excel(filename,sheet_name = sheetname)
    sheetname = '商品数据'
    itemDF = pd.read_excel(filename,sheet_name = sheetname)
 
    num = 20
    itemcf = ItemBasedCF(n_sim_item = num)
    itemcf.generate_dataset(ratingfile,itemfile,testSize = 0,lev1_ActiveScore = 3,lev2_ActiveScore = 2, lev3_ActiveScore= 1)
    itemcf.generate_itemPopular()
    itemcf.calc_item_sim_cos() 
    
    userHis_A = pd.DataFrame()
    userRec_A = pd.DataFrame()
    for ind in range(len(userDF)):
#        ind = 0
        user = userDF['用户ID'][ind]
        print('dealing with user %s '%user)
        userHis,merge_rec_1,recDF,item_TopSim = itemcf.recommendInfo(user,item = None)     
        userHis['user'] = user
        recDF['user'] = user
        userHis_A = pd.concat([userHis_A,userHis])
        userRec_A = pd.concat([userRec_A,recDF])
    userHis_A.reset_index(drop = False).to_excel(filepath + '/测试用户历史行为.xlsx')
    userRec_A.reset_index(drop = False).to_excel(filepath + '/测试用户推荐结果.xlsx')
    
    itemSim_A = pd.DataFrame()
    for ind in range(len(itemDF)):
        item = itemDF['商品ID'][ind]
        print('dealing with item %s '%item)
        itemSelfDF = itemcf.itemmap.iloc[[list(itemcf.itemmap['item']).index(item)]]
        print(itemcf.itemmap.iloc[[list(itemcf.itemmap['item']).index(item)]])
        temp = pd.DataFrame({'item':list(itemcf.item_sim_mat[item].keys()),'score':list(itemcf.item_sim_mat[item].values())}).sort_values(by = 'score',ascending = False)
        item_TopSim = pd.merge(temp.head(),itemcf.itemmap,on='item',how = 'left')
        itemSim_sub = pd.concat([itemSelfDF,item_TopSim])
        itemSim_sub['testID'] = item
        itemSim_A = pd.concat([itemSim_A,itemSim_sub])
    itemSim_A.reset_index(drop = False).to_excel(filepath + '/测试商品推荐结果.xlsx')
        
    print('finish')
    
    
    
    for num in n_sim_itemL:
        print(['*']*40)
        itemcf = ItemBasedCF(n_sim_item = num)
        itemcf.generate_dataset(ratingfile,itemfile)
        itemcf.generate_itemPopular() #不调整优先级分数 不需要更新itempopular
        itemcf.calc_item_sim_cos()
        itemcf.analyse()
###
    itemcf = ItemBasedCF()
    itemcf.generate_dataset(ratingfile,itemfile)
    itemcf.generate_itemPopular()
    itemcf.calc_item_sim_personCor()
    itemcf.analyse()


