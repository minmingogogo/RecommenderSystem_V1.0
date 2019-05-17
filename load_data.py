# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:37:47 2019

@author: shine
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import json
import re
from collections import defaultdict

####---------------------         绘图配置           -----------------------####
#plt.stytle.use('ggplot')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号            
plt.rcParams['figure.figsize'] = (10.0, 5.0) # 设置figure_size尺寸
plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
plt.rcParams['image.cmap'] = 'gray' # 设置 颜色 style
            
#figsize(12.5, 4) # 设置 figsize
plt.rcParams['savefig.dpi'] = 100 #图片像素
plt.rcParams['figure.dpi'] = 100 #分辨率
# 默认的像素：[6.0,4.0]，分辨率为100，图片尺寸为 600&400
# 指定dpi=200，图片尺寸为 1200*800
# 指定dpi=300，图片尺寸为 1800*1200
# 设置figsize可以在不改变分辨率情况下改变比例
import seaborn as sns
sns.set(font='SimHei')  # 解决Seaborn中文显示问题
import warnings
warnings.filterwarnings(action = 'ignore') #忽略警告
####---------------------         end           -----------------------####
                                           

#==============================================================================
# 1、 检查数据概况
#==============================================================================
#   导入数据
def loadData(datadir,sheetname):
    if datadir[-4:] == '.csv':
        data = pd.read_csv(datadir)
    elif datadir[-4:] == '.xls':
        data = pd.read_excel(datadir)
    elif datadir[-5:] == '.xlsx':
        data = pd.read_excel(datadir,sheet_name = sheetname)
    #   表头标题英文统一为大写
    for i in range(len(list(data.columns))):
        col = data.columns[i]
#        print(col)
        if re.findall('[a-z]',col) != []:            
            data.rename(columns = {data.columns[i]:data.columns[i].upper()},inplace =True) 
#   数据预览
    data = data.astype({'商品ID':'str','用户ID':'str'})

    print(data.info())                   
    return data
    
#   查询重复记录

def unique_count(data,unique_ColName):
#    print(unique_ColName)
#    print(type(unique_ColName))    
#    print(data[unique_ColName])
    ind_col = list(data.columns).index(unique_ColName)
    print(ind_col)
    temp = pd.DataFrame(data[unique_ColName].value_counts()).reset_index(drop = False)
    print('%s 唯一记录数 ：%s '%(unique_ColName,len(temp)))
    un_unique_len = len(temp[temp[unique_ColName] > 1 ])
    if un_unique_len > 0:    
        print('%s 重复记录数 ：%s'%(unique_ColName,un_unique_len))
        random_val = temp['index'][np.random.randint(0,un_unique_len)]
        temp = data[data[unique_ColName] == random_val]
        print('随机抽取一个重复记录 %s = %s'%(unique_ColName,random_val))
        print(temp)        
    else:
        print('%s 无重复记录'%unique_ColName)
    return '检查完毕'


#   单值分布分析
def singleFactorAnalyse(analyse_data,analyse_col):
    print(analyse_data[analyse_col].describe())
    count_df = pd.DataFrame(analyse_data[analyse_col].value_counts()).reset_index(drop = False)
    #   seaborn
    count_df.plot(kind = 'bar')
    # 设置坐标轴刻度的字体大小
    # matplotlib.axes.Axes.tick_params
    #fig = plt.figure(figsize=(10,12))
    sns.barplot(x = count_df['index'],y = count_df[analyse_col])

    count_df = pd.DataFrame(analyse_data[analyse_col].value_counts()).reset_index(drop = False)
    return count_df

def analyseData(data,unique_ColName,analyse_col):
    # unique_ColName = '行为ID'     #订单行ID
    print('【1】数据集唯一记录字段唯一性检测：') 
    print(type(data))
    print('[ %s ]唯一性检测：'%(unique_ColName))
    try:
        print(unique_count(data,unique_ColName))
    except:
        print('唯一性验证失败，需要手动执行验证代码')
    print('-'*20)  
    #   关键指标规模：用户/商品
    
    print('【2】整体数量分布:')
    print('用户数：%s'%len(pd.DataFrame(data['用户ID'].value_counts())))  
    print('商品数：%s'%len(pd.DataFrame(data['商品ID'].value_counts())))  
    print('-'*20)
    
    print('【3】用户-商品-行为分布')
    #analyse_col = '加入购物车次数'      # 数量 , 加入购物车次数
    user_item_df = singleFactorAnalyse(data,analyse_col)
    #   暂时用 times 
    user_item_count = pd.DataFrame(pd.pivot_table(data,index = ['用户ID', '商品ID'],values = analyse_col,aggfunc = np.sum)).reset_index(drop = False)
    print('用户-商品-%s 统计：')
#    print(user_item_count)
    data['times'] = 1
    user_item_times = pd.DataFrame(pd.pivot_table(data,index = ['用户ID', '商品ID'],values = 'times',aggfunc = np.sum)).reset_index(drop = False)
    
    #   user_item_count 同一用户每次购买一款商品数量分布
#    user_item_count.columns
    print('%s 95分位数：%s '%(analyse_col, np.percentile(user_item_count[analyse_col],95)))
    user_item_count_fb = singleFactorAnalyse(user_item_count,analyse_col)
    print(user_item_count_fb)
    #   user_item_times 同一用户购买一款商品的次数分布
#    user_item_times.columns
    print('次数 95分位数：%s '%np.percentile(user_item_times['times'],95))
    user_item_times_fb = singleFactorAnalyse(user_item_times,'times')
    print(user_item_times_fb)
    
    return user_item_times,user_item_count_fb,user_item_times_fb

def createTreesUser_Item_times(dataframe,lev1,lev2,val):
    num = len(dataframe)
    dataframe = dataframe.astype({lev1:'str',lev2:'str'})
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
        if ind % 1000 == 0 or ind == num-1:            
            print('已处理 %s 个 已耗时 %s 分钟'%(str(ind),str((time.clock()-s_t)/60)))
    return user_item_Dict


#==============================================================================
# 2、数据集验证及处理                       
#==============================================================================
#   配置地址：datapath: data所在文件夹路径，dataname:data文件名                       
#   unique_ColName : 唯一记录字段名
#   analyse_col : 用户-物品 单次行为统计列字段名（按数量，一个记录记为一次）
datanameL = ['电商数据汇总-20190312.xlsx']*5


sheetnameL = ['订单',
              '购物车',
              '立即购买',
              '浏览',
              '收藏'
              ]
unique_ColNameL = ['订单行ID',
                   '行为ID',
                   '行为ID',
                   '行为ID',
                   '行为ID'
                   ]
analyse_colL = ['数量',
                '加入购物车次数',
                '次数',
                '浏览次数',
                '加入收藏次数'
                  ]
nameList = ['DinDan', 'JiaGou', 'LiGou', 'LiuLan', 'ShouCang']
namemap = ['订单','加购','立购','浏览','收藏']

def done():
    datapath = 'D:\myGitHub\KYE_AI\ecomm\RecommenderSystem_V1.0\data\电商数据汇总-20190312'

#   加载数据 查看字段名
    mergeDF = pd.DataFrame()
    for ind in range(5):
        dataname = datanameL[ind]
        datadir = datapath + '/' + dataname
        sheetname = sheetnameL[ind] #   已登录数据 Sheet1  已登录用户数据
        print(ind,datadir)
        data = loadData(datadir,sheetname)
        unique_ColName = unique_ColNameL[ind]
        analyse_col = analyse_colL[ind]     
        user_item_times,user_item_count_fb,user_item_times_fb = analyseData(data,unique_ColName,analyse_col)
        dataframe = user_item_times
        lev1 = '用户ID'
        lev2 = '商品ID'
        val = 'times'
        user_item_Dict = createTreesUser_Item_times(dataframe,lev1,lev2,val)
        user_item_times['active'] = namemap[ind]
        # 导出
        filename_prefix = 'create'
        dataType = 'Dataframe'
        keyword = 'user_item_times'
        filename_suffix = nameList[ind]
        
        filename_txt = datapath + '/' + '_'.join([filename_prefix,dataType,keyword,filename_suffix]) +'.txt' 
        filename_pkl = datapath + '/' + '_'.join([filename_prefix,dataType,keyword,filename_suffix]) +'.pkl' 
        filename_excel = datapath + '/' + '_'.join([filename_prefix,dataType,keyword,filename_suffix]) +'.xlsx' 
        filename_json = datapath + '/' + '_'.join([filename_prefix,dataType,keyword,filename_suffix]) +'.json' 
             
        #   1) 数据写入 txt                                         
#        file_handle = open( filename_txt, 'w', encoding='UTF-8')
#        file_handle.write(','.join(["index"]+list(user_item_times.columns))+'\n') # 写列名
#        for ind in range(len(user_item_times)):
#            serise = ','.join([str(ind)]+ list(map(lambda x : str(x),list(user_item_times.iloc[ind,:])))) +'\n'
#            file_handle.write(serise)
#        file_handle.close()
        #   2) 数据写入excel
        user_item_times.to_excel(filename_excel) 
        mergeDF = pd.concat([mergeDF,user_item_times])
        #   3) 数据写入pkl
#        dataout = open(filename_json,'wb')
#        pkl.dump((user_item_times),dataout,protocol=2)
#        dataout.close()
        
        #   4) dict 数据写入 json
        with open(filename_json,'w') as f:
            json.dump(user_item_Dict,f)
        print('finish dump %s'%filename_json)
    mergeDF.reset_index(drop = True,inplace= True)
    mergeDF.to_excel(datapath + '/' + '_'.join([filename_prefix,dataType,keyword,'Merge']) +'.xlsx')
    return 'finish'

done()
  
    datapath = 'D:\myGitHub\KYE_AI\ecomm\RecommenderSystem_V1.0\data'
    dataname = '商品浏览数据.xlsx'
    datadir = datapath + '/' + dataname
    sheetname = 'Sheet1' #   已登录数据 Sheet1  已登录用户数据
#   加载数据 查看字段名
    data = loadData(datadir,sheetname)

unique_ColName = '浏览行为ID'
analyse_col = '浏览次数'
user_item_times,user_item_count_fb,user_item_times_fb = analyseData(data,unique_ColName,analyse_col)

dataframe = user_item_times
lev1 = '用户ID'
lev2 = '商品ID'
val = 'times'
user_item_Dict = createTreesUser_Item_times(dataframe,lev1,lev2,val)


#==============================================================================
# 3、生成评分数据集
#==============================================================================

#   1）create userID corresponding table
#   2) create itemID corresponding table
#   3) create user-item-rating table
    '''
    import pickle as pkl
    outdir = datapath + '/create_user_item_times.pkl'
    dataout = open(outdir,'wb')
    pkl.dump((user_item_times),dataout,protocol=2)
    dataout.close()  

    #   读 
    file_path = "abc.csv"
    with open(file_path) as f:
        for i, line in enumerate(f.readlines()):
            # i 是行数
            # line 是每行的字符串，可以使用 line.strip().split(",") 按分隔符分割字符串
        f.close()    
        
    '''
import json
#   写
filename_prefix = 'create'
dataType = 'Dataframe'
keyword = 'user_item_times'
filename_suffix = 'LiuLan'

filename_txt = datapath + '/' + '_'.join([filename_prefix,dataType,keyword,filename_suffix]) +'.txt' 
filename_pkl = datapath + '/' + '_'.join([filename_prefix,dataType,keyword,filename_suffix]) +'.pkl' 
filename_excel = datapath + '/' + '_'.join([filename_prefix,dataType,keyword,filename_suffix]) +'.xlsx' 
filename_json = datapath + '/' + '_'.join([filename_prefix,dataType,keyword,filename_suffix]) +'.json' 
     
#   1) 数据写入 txt                                         
file_handle = open( filename_txt, 'w', encoding='UTF-8')
#file_handle.write("index"+","+"Kernel"+“,"+"Context"+","+"Stream"+'\n') # 写列名
file_handle.write(','.join(["index"]+list(user_item_times.columns))+'\n') # 写列名
for ind in range(len(user_item_times)):
#    ind = 0
    serise = ','.join([str(ind)]+ list(map(lambda x : str(x),list(user_item_times.iloc[ind,:])))) +'\n'
    file_handle.write(serise)
#serise = str(i)+","+kernel_name_new+","+context+","+stream  # 每个元素都是字符串，使用逗号分割拼接成一个字符串
#file_handle.write(serise+'\n') # 末尾使用换行分割每一行。
file_handle.close()
#   2) 数据写入excel
user_item_times.to_excel(filename_excel) 

#   3) 数据写入pkl
dataout = open(filename_json,'wb')
pkl.dump((user_item_times),dataout,protocol=2)
dataout.close()

#   4) dict 数据写入 json
with open(filename_json,'w') as f:
    json.dump(user_item_Dict,f)
print('finish dump %s'%filename)



#==============================================================================
# 4、数据合并
#==============================================================================

    #   读 
import os
filelist = os.listdir(datapath)
fileList = [x for x in filelist if x.find('.txt')!=-1]
dfname_List = list(map(lambda x : re.sub('.txt','',x.split('_')[-1]),fileList))


for ind in range(len(fileList)):
#    ind = 1
    name = dfname_List[ind]
    print('%s %s'%(ind,name))
    locals()[name] = pd.DataFrame()
    file_path = datapath + '/' + fileList[ind]
    with open(file_path) as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            index, user, item, times = line.strip('\r\n').split(',')
            subdf = pd.DataFrame({'user':[user],'item':[item],'times':[times]})
            if i % 10000 == 0:
                print('已处理 %s '%i)
            locals()[name] = pd.concat([locals()[name],subdf])
        locals()[name].reset_index(drop = True , inplace = True)
            # i 是行数
            # line 是每行的字符串，可以使用 line.strip().split(",") 按分隔符分割字符串
        print('%s 数据量 %s '%(name,len(locals()[name])))
        locals()[name].head()
        f.close()    
    
JiaGou.head()
nameList = ['DinDan', 'JiaGou', 'LiGou', 'LiuLan', 'ShouCang']
namemap = ['订单','加购','立购','浏览','收藏']
for ind in range(len(fileList)):
    name = nameList[ind]
    colval = namemap[ind]
    locals()[name]['active'] = colval

JiaGou.head()


mergeDf = pd.DataFrame()
for ind in range(len(fileList)):
    name = nameList[ind]
    mergeDf = pd.concat([mergeDf,locals()[name]])
mergeDf.reset_index(drop = True,inplace = True)
len(mergeDf)

#   写
user_item_times = mergeDf
file_handle = open( datapath +'/create_Dataframe_user_item_times_Merge.txt', 'w', encoding='UTF-8')
#file_handle.write("index"+","+"Kernel"+“,"+"Context"+","+"Stream"+'\n') # 写列名
file_handle.write(','.join(["index"]+list(user_item_times.columns))+'\n') # 写列名
for ind in range(len(user_item_times)):
#    ind = 0
    serise = ','.join([str(ind)]+ list(map(lambda x : str(x),list(user_item_times.iloc[ind,:])))) +'\n'
    file_handle.write(serise)
#serise = str(i)+","+kernel_name_new+","+context+","+stream  # 每个元素都是字符串，使用逗号分割拼接成一个字符串
#file_handle.write(serise+'\n') # 末尾使用换行分割每一行。
file_handle.close()
user_item_times.to_excel(datapath +'/create_Dataframe_user_item_times_Merge.xlsx') 

#   dict 数据写入 json
dataframe = mergeDf
lev1 = 'user'
lev2 = 'item'
val = 'times'
user_item_Dict = createTreesUser_Item_times(dataframe,lev1,lev2,val)

filename = 'create_Dict_user_item_times_Merge.json'
with open(datapath + "/" + filename,'w') as f:
    json.dump(user_item_Dict,f)
print('finish dump %s'%filename)


mergeDf['active'].value_counts()

