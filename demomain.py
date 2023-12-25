if __name__ == '__main__':
    '''
    下面代码为定式,不需要动
    '''
    a = 'mydata'
    b = '123'
    start_time = time.time()
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True 
    config = Config(a,b)
    vocab_path= 'mydata/data/vocab3.pkl'
    vocab = pkl.load(open(vocab_path, 'rb'))
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    config.n_vocab = len(vocab)
    global cfggb
    cfggb = len(vocab)
    model = Model(config).to('cpu')
    model.load_state_dict(torch.load(config.save_path))





    text = ""           #在这里输入评论
    predict(model,text) #推导函数,输入(model,text),输出为置信度最高的两种情感