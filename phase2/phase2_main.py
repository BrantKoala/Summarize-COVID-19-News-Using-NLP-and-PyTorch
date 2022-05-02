from __future__ import unicode_literals, print_function, division

import random
import time
from io import open

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 601
input_lang, output_lang, pairs = None, None, None
teacher_forcing_ratio = 0.5


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split():
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# clean up
# disregard any input beyond the length
def select_short(dataset, target, max_len_text=600, max_len_target=30):
    short_text = []
    short_summary = []
    for i in range(len(dataset)):
        if (len(target[i].split()) <= max_len_target and len(dataset[i].split()) <= max_len_text):
            short_text.append(dataset[i])
            short_summary.append(target[i])
    return pd.DataFrame({'text': short_text, 'summary': short_summary})


def readData(text, summary):
    print("Reading lines...")
    # Split every line into pairs and normalize
    pairs = [[text[i], summary[i]] for i in range(len(text))]
    input_lang = Lang(text)
    output_lang = Lang(summary)
    return input_lang, output_lang, pairs


def prepareData(lang1, lang2):
    input_lang, output_lang, pairs = readData(lang1, lang2)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs


def max_sequence(sequence):
    seq_len = []
    i = 1
    for one_seq in sequence:
        seq_spl = one_seq.split()
        seq_len.append(len(seq_spl))
        i = i + 1
    max_seq = max(seq_len)
    return max_seq


'''
Encoder是个RNN，它会遍历输入的每一个Token(词)，每个时刻的输入是上一个时刻的隐状态和输入，然后会有一个输出和新的隐状态。
这个新的隐状态会作为下一个时刻的输入隐状态。每个时刻都有一个输出，对于seq2seq模型来说，我们通常只保留最后一个时刻的隐状态，
认为它编码了整个句子的语义，但是后面我们会用到Attention机制，它还会用到Encoder每个时刻的输出。
Encoder处理结束后会把最后一个时刻的隐状态作为Decoder的初始隐状态。
'''

'''
seq2seq模型常用于机器翻译，由两部分组成：encoder和decoder，一般使用RNN网络实现，比较常用的就是LSTM和GRU了。机器翻译时，
encoder作用是对输入句子进行特征提取，它的最后一个输出就是从这句话捕获的最后的特征。decoder就利用编码器最后的输出特征解码成目标语言。
Seq2Seq模型有一个缺点就是句子太长的话encoder会遗忘，那么decoder接受到的句子特征也就不完全，因此引入attention机制，
Decoder每次更新状态的时候都会再看一遍encoder所有状态，还会告诉decoder要更关注哪部分，这样能大大提高翻译精度。
具体实现就是不再单纯使用encoder的最后一个状态进行解码，在encoder工作时会保存其每一个输出和隐藏状态，
在解码时用于计算当前时刻隐藏状态与编码所有时刻隐藏状态的相关性。
'''


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        # 继承 nn.Module 的神经网络模块在实现自己的 __init__ 函数时，一定要先调用。只有这样才能正确地初始化自定义的神经网络模块
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # 定义一个具有input_size个单词，维度为hidden_size的查询矩阵
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 输入hidden_size个特征维度 隐藏是hidden_size个特征维度
        # 门：这里GRU的hide layer维度和embedding维度一样，但并不是必须的
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # (seq_len, batch, hidden_size)，view实际上是对现有tensor 改造的方法
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        # 获取每个GRU的输出和隐藏状态，用于后续计算attention
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


'''
注意力机制允许解码器针对自身输出的每一步都“关注”编码器输出的不同部分。首先需要计算一组注意力权重，
用来乘以编码器输出向量实现加权生成注意力向量。然后将此注意力向量与解码器当前输入进行拼接作为GRU的输入：
'''


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # 全连接层
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # 先把输入embedding
        embedded = self.embedding(input).view(1, 1, -1)
        # dropout防止过拟合(丢弃正则化)
        embedded = self.dropout(embedded)
        # 矩阵相乘，用注意力权重乘以编码输出
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # 将输入的embedding层和注意力层拼接，按维数1拼接（横着拼）
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # 拼好后加个全连接层然后解压缩维度0。
        output = self.attn_combine(output).unsqueeze(0)
        # 激活函数
        output = F.relu(output)
        # 输入GRU
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split()]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


'''
为了训练，我们通过编码器运行输入句子，并跟踪每个输出和最新的隐藏状态。然后，解码器将令牌作为其第一个输入，
并将编码器的最后一个隐藏状态作为其第一个隐藏状态。<SOS>

“教师强制”是使用真实目标输出作为每个下一个输入的概念，而不是使用解码器的猜测作为下一个输入。使用教师强制会导致它更快地收敛，
但是当训练有素的网络被利用时，它可能会表现出不稳定。

你可以观察教师强迫的网络的输出，这些网络以连贯的语法阅读，但远离正确的翻译 - 直观地说，它已经学会了表示输出语法，
一旦老师告诉它前几个单词，它就可以“拾取”含义，但它还没有正确地学会如何从翻译中创建句子。

由于PyTorch的自动评分给我们的自由，我们可以随机选择使用或不使用简单的if语句来强制使用教师强迫。打开以使用更多。
'''


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    # 初始化编码器的隐藏状态
    encoder_hidden = encoder.initHidden()
    # 清除渐变。Pytorch在随后的反向传播中积累梯度，因此您需要在开始训练之前清除它。否则，梯度计算将是错误的。
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # 从相应的张量中找出输入和目标长度
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    # 将encoder_outputs初始化为max_length大小的割炬阵列，并用零填充它。.我们将在为每个输入生成编码器输出时更新此数组。
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    # 将损失初始化为零 。我们将在训练时更新此损失，并在后续步骤中对其运行反向传播。
    loss = 0
    print("input_length={}".format(input_length))
    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
        # print(encoder_output.size())
        encoder_outputs[i] = encoder_output[0, 0]
    # 定义解码器输入和解码器隐藏状态。最初，SOS_token是句子的开头标记。
    decoder_input = torch.tensor([[SOS_token]], device=device)
    # 首字母将初始化为decoder_hidden
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    # 反向传播
    loss.backward()
    # 取编码器和解码器的梯度下降
    encoder_optimizer.step()
    decoder_optimizer.step()
    # 回波损耗
    return loss.item() / target_length


'''
整个训练过程如下所示：
启动计时器
初始化优化程序和条件
创建训练对集
启动空损耗数组进行绘图
然后我们多次调用，偶尔打印进度（示例百分比，到目前为止的时间，估计的时间）和平均损失。
'''


def trainIters(encoder, decoder, n_iters, learning_rate=0.01):
    print("Training....")
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # TODO:这里可能要改进以避免重复抽取
    training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        if iter % 1000 == 0:
            print(iter, "/", n_iters + 1)
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss


def infer(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        output_words, attentions = infer(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        with open('evaluation_input.txt', 'a') as out:
            out.write('{}, {}\n'.format(pair[1], output_sentence))


def main():
    df = pd.read_csv("skimmed_news.csv", encoding="utf-8")
    global input_lang, output_lang, pairs
    input_lang, output_lang, pairs = prepareData(list(df['text']), list(df['summary']))
    length_result = []
    for pair in pairs:
        length_result.append(len(pair[0].split()))
    print("max(length_result)={}".format(max(length_result)))
    # 训练
    hidden_size = 300
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    trainIters(encoder, attn_decoder, 1000)
    ENCODER_MODEL_PATH = 'encoder_model.pt'
    DECODER_MODEL_PATH = 'decoder_model.pt'
    torch.save(encoder.state_dict(), ENCODER_MODEL_PATH)
    torch.save(attn_decoder.state_dict(), DECODER_MODEL_PATH)
    evaluateRandomly(encoder, attn_decoder)


if __name__ == "__main__":
    main()
