import pandas as pd
import numpy as np


def get_test_data(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    df.head()

    geohasd_df_dict = {}
    date_df_dict = {}
    number_hash = 0
    number_date = 0

    # 循环遍历节点数据的 "geohash_id" "date_id"列，将不同的节点ID和时间映射到数字
    for i in df["geohash_id"]:

        if i not in geohasd_df_dict.keys():
            geohasd_df_dict[i] = number_hash
            number_hash += 1

    for i in df["date_id"]:
        if i not in date_df_dict.keys():
            date_df_dict[i] = number_date
            number_date += 1

    print(geohasd_df_dict)
    print(date_df_dict)

    # number_hash 1140
    # number_date 90
    # 创建了一个二维列表 new_data，其中所有元素都初始化为0
    # new_data 的维度取决于节点ID和日期ID的数量
    new_data = np.zeros((len(date_df_dict), len(geohasd_df_dict),36),dtype=float)
    for index, row in df.iterrows():
        # print(index)
        hash_index, date_index = geohasd_df_dict[row["geohash_id"]], date_df_dict[row["date_id"]]
        # 将时间index加到里面

        # [date_index] + list(row.iloc[2:]) 是一个列表，它将日期索引作为第一个元素，然后将 row 数据中从第三个元素开始的所有元素添加到列表中。
        # 这相当于将日期和节点特征数据合并为一个列表
        new_data[date_index][hash_index] = [date_index] + list(row.iloc[2:])
    """
    new_data 是一个二维列表。
    每行代表一个日期，每列代表一个节点。
    每个元素是一个列表，其中包含日期ID和节点的特征数据。
    """
    new_data = np.array(new_data)
    return new_data


def distEclud(node1, node2):
    """
    :param node1: 节点1不同date的数据
    :param node2: 节点2不同date的数据
    :return: 两个节点的相似度
    """
    # node1和node2为该
    dist = 0
    for date_index in range(node1.shape[0]):
        dist_date = 0
        # 使用zip函数同时迭代两个列表
        for i, j in zip(node1[date_index][1:], node2[date_index][1:]):
            # 计算对应元素的差值，然后求和
            dist_date += abs(i - j)
        # 计算平均差值
        dist_date = dist_date / (len(node1[0]) - 1)
        dist += dist_date
    dist = dist / (node1.shape[0])
    return dist


def closest_node(all_nodes, node):
    """
    :param all_nodes:所有节点的数据
    :param node: 源节点index
    :return: 距离源节点最近的节点数据
    """
    closest_node_index = 0
    closest_node_dist = distEclud(all_nodes[:, closest_node_index], all_nodes[:, node])

    # 遍历所有节点（除了源节点）
    for curr_node in range(all_nodes.shape[1]):
        if curr_node != node:
            curr_node_dist = distEclud(all_nodes[:, curr_node], all_nodes[:, node])
            if curr_node_dist < closest_node_dist:
                closest_node_index = curr_node
                closest_node_dist = curr_node_dist

    return closest_node_index

def process_zero(node_index, all_nodes_data):
    """
    :param data: 同一节点不同date的数据
    :param all_nodes_data: 所有节点所有date的数据
    :return:
    """
    data = all_nodes_data[:, node_index]
    for j in range(1, len(data[0])):
        # len(data[0])为列表长度，列表可能由日期索引+元素特征值+预测指数构成

        # feature_date为当前节点第j个特征不同date的列表，转换成列表便于判断0值数量
        feature_data = []
        num_zero = 0
        for i in range(data.shape[0]):
            # data.shape[0]为date数
            if data[i][j] == 0:
                num_zero += 1
            feature_data.append(data[i][j])

        if num_zero == data.shape[0]:
            # 全都是0，寻找最接近的点
            closest_node_index = closest_node(all_nodes_data, node_index)
            closest_node_data = all_nodes_data[:, closest_node_index]
            # print(j, i, "最近的点是", closest_node_index)
            # print(closest_node_data)

            for index in range(len(feature_data)):
                if feature_data[index] == 0:
                    data[index][j] = closest_node_data[index][j]
        elif num_zero != 0:
            # 用其他非零值的平均值填充
            # 计算所有非零值的平均值
            average = sum(num for num in feature_data if num != 0) / (len(feature_data)-num_zero) if feature_data else 0
            # 将所有的零值替换为平均值
            for index in range(len(feature_data)):
                if feature_data[index] == 0:
                    feature_data[index] = average
    return data



if __name__ == "__main__":
    data = get_test_data('./dataset/node_test_3_B.csv')
    for node in range(data.shape[1]):
        process_zero(node, data)
    print(data)
