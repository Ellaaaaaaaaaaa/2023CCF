import pandas as pd
import numpy as np
# 读取CSV文件
import my_model
import  torch
import argparse
from tqdm import tqdm, trange
import data
import torch.nn as nn
#获得训练集

criterion = nn.MSELoss()  # 使用均方误差损失函数计算MSE
def get_train_data(file_path,edge_pth):
    df = pd.read_csv(file_path, encoding='utf-8')
    edge_df = pd.read_csv(edge_pth, encoding='utf-8')
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

    # number_hash 1140
    # number_date 90
    # 创建了一个二维列表 new_data，其中所有元素都初始化为0
    # new_data 的维度取决于节点ID和日期ID的数量
    new_data = [len(geohasd_df_dict) * [0]] * len(date_df_dict)
    for index, row in df.iterrows():
        # print(index)
        hash_index, date_index = geohasd_df_dict[row["geohash_id"]], date_df_dict[row["date_id"]]
        #将时间index加到里面

        # [date_index] + list(row.iloc[2:]) 是一个列表，它将日期索引作为第一个元素，然后将 row 数据中从第三个元素开始的所有元素添加到列表中。
        # 这相当于将日期和节点特征数据合并为一个列表
        new_data[date_index][hash_index] = [date_index]+list(row.iloc[2:])
    """
    new_data 是一个二维列表。
    每行代表一个日期，每列代表一个节点。
    每个元素是一个列表，其中包含日期ID和节点的特征数据。
    """
    new_data = np.array(new_data)
    # new_data.shape 90 1140 38
    # x_train,y_train = new_data[:, :-2], new_data[:, -2:]
    # print(len(geohasd_df_dict))
    # exit()
    # print(x_train.shape)
    # print(y_train.shape)
    #这里构建邻接矩阵其中mask表示1为有边，0无边， value_mask表示有值
    #并且这里我考虑mask是一个无向图，如果有向删除x_mask[date_index][point2_index][point1_index],value_mask同理
    # todo 这里源代码考虑为无向图，是否考虑边的方向？不过我个人感觉先不改这里
    x_mask =  np.zeros((len(date_df_dict),len(geohasd_df_dict),len(geohasd_df_dict),1), dtype = float)
    x_edge_df =np.zeros((len(date_df_dict),len(geohasd_df_dict),len(geohasd_df_dict),2), dtype = float)

    # x_mask 中的值为1表示存在边，类似邻接矩阵
    # x_edge_df 中的值包含了边的特征信息
    for index, row in edge_df.iterrows():
        # print(index)
        if row["geohash6_point1"] not in geohasd_df_dict.keys() or row["geohash6_point2"] not in geohasd_df_dict.keys():
            continue
        point1_index,point2_index,F_1,F_2,date_index= geohasd_df_dict[row["geohash6_point1"]],geohasd_df_dict[row["geohash6_point2"]]\
            ,row["F_1"],row["F_2"],date_df_dict[row["date_id"]]
        x_mask[date_index][point1_index][point2_index] = 1
        x_mask[date_index][point2_index][point1_index] = 1
        # TODO 这里是直接输入边特征的，数据没处理 对数处理
        x_edge_df[date_index][point1_index][point2_index] =  [F_1,F_2]
        x_edge_df[date_index][point2_index][point1_index] = [F_1, F_2]
    # print(data)

    return    geohasd_df_dict, date_df_dict, new_data,x_mask, x_edge_df

def eval(model, dataset, args):
    model.eval()
    with torch.no_grad():

        dev_loss = 0.0
        for j in trange(dataset.batch_count):
            x_date, x_feature, x_mask_data, x_edge_data, x_tags = dataset.get_batch(j)
            act_pre, con_pre = model(x_date, x_feature, x_mask_data)
            predict = torch.cat((act_pre, con_pre), dim=-1)
            loss = criterion(predict, x_tags)
            dev_loss+= loss
        print("this epoch dev loss is {}".format(dev_loss))
        model.train()


def predict(model, dataset, args, geohasd_df_dict,date_df_dict_test):
    model.eval()
    with torch.no_grad():
        predictions = []
        for j in trange(dataset.batch_count):
            # 获取测试集的一个批次数据
            x_date, x_feature, x_mask_data, x_edge_data= dataset.get_batch(j)
            # 进行预测
            act_pre, con_pre = model(x_date, x_feature, x_mask_data)
            # 将预测结果整理为需要的格式
            # act_pre (4,1140,1)
            for idx in range(act_pre.size(1)):
                geohash_id = list(geohasd_df_dict.keys())[idx]
                date_ids = list(date_df_dict_test.keys())

                for date_id_idx in range(len(date_ids)):
                    date_id = date_ids[date_id_idx]
                    act_level = max(0, act_pre[date_id_idx, idx, 0].cpu().item())
                    con_level = max(0, con_pre[date_id_idx, idx, 0].cpu().item())

                    prediction = {
                        "geohash_id": geohash_id,
                        "activity_level": act_level,
                        "consumption_level": con_level,
                        "date_id": date_id
                    }
                    predictions.append(prediction)

    # 将所有预测结果整理为 DataFrame
    predictions_df = pd.DataFrame(predictions)

    return predictions_df



def train(args):

    geohasd_df_dict, date_df_dict, x_train, x_mask, x_edge_df = get_train_data('./dataset/train_90.csv',
                                                                                        "./dataset/edge_90.csv")
    # 分割各种训练集测试集
    x_train,x_dev = torch.tensor(x_train[:int(len(x_train)*args.rat)]),torch.tensor(x_train[int(len(x_train)*args.rat):])
    x_mask_train,x_mask_dev = torch.tensor(x_mask[:int(len(x_mask)*args.rat)]),torch.tensor(x_mask[int(len(x_mask)*args.rat):])
    x_edge_train, x_edge_dev = torch.tensor(x_edge_df[:int(len(x_edge_df) * args.rat)]),torch.tensor( x_edge_df[int(len(x_edge_df) * args.rat):])

    # 日期的嵌入维度
    date_emb = 5
     # 这里的x包含了date_id+F35个特征+2个y值的
    # train_activate = torch.tensor(y_train[:, -2])
    # train_consume = torch.tensor(y_train[:, -1])


    # rmse_loss = torch.sqrt(mse_loss)
    # TODO 发现这里没有结合GAT和Bi-LSTM
    # TODO 要调一下这里模型的参数
    # model = my_model.GAT(date_emb =[len(date_df_dict),date_emb], nfeat=35, nhid=64, dropout=0.3, alpha=0.3, nheads=8).to(args.device)
    # 定义 BiLSTM 模型
    # bilstm_model = my_model.BiLSTM(input_size=64, hidden_size=64, output_size=2,num_layers=2, dropout=0.3).to(args.device)
    model = my_model.BILSTM(date_emb =[len(date_df_dict),date_emb], nfeat=35, nhid=64, dropout=0.3, alpha=0.3, nheads=8).to(args.device)
    optimizer = torch.optim.Adam(params=model.parameters(),lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decline, gamma=0.8, last_epoch=-1)
    model.train()
    trainset = data.DataIterator(x_train,x_mask_train,x_edge_train, args)
    valset =data.DataIterator(x_dev,x_mask_dev,x_edge_dev, args)

    for indx in range(args.epochs):
        train_all_loss = 0.0
        for j in trange(trainset.batch_count):
            # todo x_edge_data即边上特征值组成的数据没使用到
            x_date,x_feature,x_mask_data,x_edge_data,x_tags= trainset.get_batch(j)
            # torch.Size([4, 1140])torch.Size([4, 1140, 35])torch.Size([4, 1140, 1140, 1])torch.Size([4, 1140, 1140, 2])torch.Size([4, 1140, 2])
            # todo nhid隐藏层输入的是边的邻接矩阵关系x_mask_data，这里没有考虑边上的值
            act_pre, con_pre= model(x_date,x_feature,x_mask_data)
            # act_pre, con_pre= bilstm_model(gat_node_features)
            # 得到活跃指数和消费指数的预测结果，并将它们拼接在一起
            predict = torch.cat((act_pre, con_pre), dim=-1)
            # ([4, 1140, 2])
            loss = criterion(predict, x_tags)
            train_all_loss += loss
            optimizer.zero_grad()
            loss.backward()
        optimizer.step()
        print('this epoch train loss :{0}'.format(train_all_loss))
        # scheduler.step()
        eval(model,valset, args)


    # 在训练循环结束后，保存模型参数
    torch.save(model.state_dict(), 'LSTM_500_0.005_4.pth')




def test(args):
    geohasd_df_dict_test, date_df_dict_test, x_test, x_mask_test, x_edge_test = get_train_data(
        './dataset/node_test_4_A.csv',
        './dataset/edge_test_4_A.csv')

    # 日期的嵌入维度
    date_emb = 5
    model = my_model.GAT(date_emb=[90, date_emb], nfeat=35, nhid=64, dropout=0.3, alpha=0.3,
                         nheads=8).to(args.device)
    # 转换为 torch.Tensor
    x_test, x_mask_test, x_edge_test = torch.tensor(x_test), torch.tensor(x_mask_test), torch.tensor(x_edge_test)
    testset = data.DataIteratorTest(x_test, x_mask_test, x_edge_test, args)
    # 载入模型参数
    model.load_state_dict(torch.load('GAT_500_0.005_16.pth'))
    # 在测试集上进行预测
    predictions_df = predict(model, testset, args, geohasd_df_dict_test, date_df_dict_test)

    # 将预测结果保存到 CSV 文件
    predictions_df.to_csv("predictions_test_4_A.csv", sep='\t', index=False)


if __name__ == "__main__":

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500,
                        help='training epoch number')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size')
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')
    parser.add_argument('--lr', type=float, default=5e-3,
                        )
    parser.add_argument('--rat', type=float, default=0.9,)

    parser.add_argument('--decline', type=int, default=30, help="number of epochs to decline")
    train(parser.parse_args())
    # test(parser.parse_args())




