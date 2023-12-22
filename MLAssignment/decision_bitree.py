from splitInfo import info_entropy, gini_index, split_samples, sum_of_each_label
import datetime

# 个人信息
student_id = "2010040116"
name = "zcg" 

# 获取系统时间
current_time = datetime.datetime.now()

class biTree_node:
    '''
    二叉树节点
    '''
    def __init__(self, f=-1, fvalue=None, leafLabel=None, l=None, r=None, splitInfo="gini"):
        '''
        类初始化函数
        para f: int,切分的特征，用样本中的特征次序表示
        para fvalue: float or int，切分特征的决策值
        para leafLable: int,叶节点的标签
        para l: biTree_node指针，内部节点的左子树
        para r: biTree_node指针，内部节点的右子树
        para splitInfo="gini": string, 切分的标准，可取值'infogain'和'gini'，分别表示信息增益和基尼指数
        '''
        self.f = f
        self.fvalue = fvalue
        self.leafLabel = leafLabel
        self.l = l
        self.r = r
        self.splitInfo = splitInfo
        
def build_biTree(samples, splitInfo="gini"):
    '''构建树
    para samples：list,样本的列表，每样本也是一个列表，样本的最后一项为label，其它项为特征
    para splitInfo="gini": string, 切分的标准，可取值'infogain'和'gini'，分别表示信息增益和基尼指数
    return biTree_node:Class biTree_node,二叉决策树的根结点
    '''
    if len(samples) == 0:
        return biTree_node()
    if splitInfo != "gini" and splitInfo != "infogain":
        return biTree_node()
    
    bestInfo = 0.0
    bestF = None
    bestFvalue = None
    bestlson = None
    bestrson = None

    if splitInfo == "gini":
        curInfo = gini_index(samples) # 当前集合的基尼指数
    else:
        curInfo = info_entropy(samples) # 当前集合的信息熵
        
    sumOfFeatures = len(samples[0]) - 1 # 样本中特征的个数
    for f in range(0, sumOfFeatures): # 遍历每个特征
        featureValues = [sample[f] for sample in samples]
        for fvalue in featureValues:  # 遍历当前特征的每个值
            lson, rson = split_samples(samples, f, fvalue)
            if splitInfo == "gini":
                # 计算分裂后两个集合的基尼指数
                info = (gini_index(lson)*len(lson) + gini_index(rson)*len(rson))/len(samples)
            else:
                # 计算分裂后两个集合的信息熵
                info = (info_entropy(lson)*len(lson) + info_entropy(rson)*len(rson))/len(samples)
            gain = curInfo - info # 计算基尼指数减少量或信息增益
            # 能够找到最好的切分特征及其决策值，左、右子树为空说明是叶子节点
            if gain > bestInfo and len(lson)>0 and len(rson)>0:
                bestInfo = gain
                bestF = f
                bestFvalue = fvalue
                bestlson = lson
                bestrson = rson
    
    if bestInfo > 0.0: 
        l = build_biTree(bestlson)
        r = build_biTree(bestrson)
        return biTree_node(f=bestF, fvalue=bestFvalue, l=l, r=r, splitInfo=splitInfo)
    else: # 如果bestInfo==0.0，说明没有切分方法使集合的基尼指数或信息熵下降了
        label_counts = sum_of_each_label(samples)
        # 返回该集合中最多的类别作为叶子节点的标签
        return biTree_node(leafLabel=max(label_counts, key=label_counts.get), splitInfo=splitInfo)

def predict(sample, tree):
    '''
    对样本sample进行预测
    para sample:list,需要预测的样本
    para tree:biTree_node,构建好的分类树
    return: biTree_node.leafLabel,所属的类别
    '''
    # 1、只是树根
    if tree.leafLabel != None:
        return tree.leafLabel
    else:
    # 2、有左右子树
        sampleValue = sample[tree.f]
        branch = None
        if sampleValue >= tree.fvalue:
            branch = tree.r
        else:
            branch = tree.l
        return predict(sample, branch)

def print_tree(tree, level='0'):
    '''简单打印一颗树的结构
    para tree:biTree_node,树的根结点
    para level='0':str, 节点在树中的位置，用一串字符串表示，0表示根节点，0L表示根节点的左孩子，0R表示根节点的右孩子  
    '''
    if tree.leafLabel != None:
        print('*' + level + '-' + str(tree.leafLabel)) #叶子节点用*表示，并打印出标签
    else:
        print('+' + level + '-' + str(tree.f) + '-' + str(tree.fvalue)) #中间节点用+表示，并打印出特征编号及其划分值
        print_tree(tree.l, level+'L')
        print_tree(tree.r, level+'R')

if __name__ == "__main__":
    
    # 表3-1 某人相亲数据
    blind_date = [[25, 170, 0, 20000, 1],
                  [28, 178, 1, 15000, 0],
                  [26, 165, 0, 25000, 0],
                  [30, 173, 2, 20000, 0],
                  [24, 169, 0, 10000, 0],
                  [26, 167, 1, 18000, 1],
                  [20, 169, 0, 8000, 0],
                  [23, 158, 0, 10000, 0],
                  [22, 172, 0, 12000, 1],
                  [25, 175, 0, 14000, 1],
                  [20, 160, 0, 10000, 1]]
    # print("信息增益二叉树：")
    tree = build_biTree(blind_date, splitInfo="infogain")
    # print_tree(tree)
    print('信息增益二叉树对样本进行预测的结果：')
    test_sample = [[26, 170, 2, 17000],
                   [29, 180, 1, 25000],
                   [29, 157, 0, 13000],
                   [20, 159, 0, 10000]]
    for x in test_sample:
        print(predict(x, tree))

    # print("基尼指数二叉树：")
    tree = build_biTree(blind_date, splitInfo="gini")
    # print_tree(tree)
    print('基尼指数二叉树对样本进行预测的结果：')
    test_sample = [[26, 170, 2, 17000],
                   [29, 180, 1, 25000],
                   [29, 157, 0, 13000],
                   [20, 159, 0, 10000]]
    for x in test_sample:
        print(predict(x, tree))
        
    # 输出个人信息和系统时间
    print(f"Student ID: {student_id}")
    print(f"Name: {name}")
    print(f"Current Time: {current_time}")