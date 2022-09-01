import paddle
import os


# 保存模型参数
def save_model(model, model_name):
    print('{} model saving...'.format(model_name))
    paddle.save(model.state_dict(), './save_model/{}.pdparams'.format(model_name))


# 读取模型参数
def load_model(model, model_name):
    if os.path.exists('./save_model/{}.pdparams'.format(model_name)) == False:
        print('No {} model pdparams...'.format(model_name))
    else:
        model.set_state_dict(paddle.load('./save_model/{}.pdparams'.format(model_name)))
        print('success loading {} model pdparams'.format(model_name))


# 保存指标列表
def save_miou_loss(data_list, name):
    with open(name + '.txt', 'a') as f:
        for data in data_list:
            f.write(str(data) + '\n')


# 读取保存在文件的指标
def read_miou_loss(name):
    data_list = []
    with open(name + '.txt', 'r') as f:
        for data in f.readlines():
            data_list.append(eval(data.strip()))
    return data_list
