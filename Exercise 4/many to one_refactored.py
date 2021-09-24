import torch
import torch.nn as nn
import random as rd


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size, out_encoding_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128 * 4, out_encoding_size)  # 128 is the state size # TODO 128 -> 128 * 4 (for batch?)

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 4, 128)  # Shape: (number of layers, batch size, state size) # TODO Batch size
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        # print("logits", x.shape)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        # print("out", out.shape)  # logits  torch.Size([7, 4,  13])
        reshout = out.reshape(-1, 128 * 4)  # out     torch.Size([7, 4, 128]) # TODO 128 -> 128 * 4 (for batch?)
        # print("reshout", reshout.shape)  # reshout torch.Size([28, 128])
        dense = self.dense(reshout)  # dense   torch.Size([28,   7])
        # dense.reshape(7, -1)
        # print("dense", dense.shape)  # dense of wrong shape?
        return dense

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x,
             y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        temp = nn.functional.cross_entropy(self.logits(x), y.argmax(1))
        # print("temp over")
        return temp


char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'a'  0
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 't'  1
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'c'  2
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'h'  3
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'r'  4
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 'f'  5
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 'l'  6
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'm'  7
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # 'p'  8
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # 's'  9
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],  # 'o' 10
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],  # 'n' 11
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],  # ' ' 12
]

out_encodings = [
    [1., 0., 0., 0., 0., 0., 0.],  # 'hat '
    [0., 1., 0., 0., 0., 0., 0.],  # 'rat '
    [0., 0., 1., 0., 0., 0., 0.],  # 'cat '
    [0., 0., 0., 1., 0., 0., 0.],  # 'flat'
    [0., 0., 0., 0., 1., 0., 0.],  # 'matt'
    [0., 0., 0., 0., 0., 1., 0.],  # 'cap '
    [0., 0., 0., 0., 0., 0., 1.],  # 'son '
]

encoding_size = len(char_encodings)
out_encoding_size = len(out_encodings)

# print(encoding_size)
# print(out_encoding_size)

emojis = ['🎩', '🐀', '🐈', '🏢', '👨', '🧢', '👦']
# emojis = ['hat ', 'rat ', 'cat ', 'flat', 'matt', 'cap ', 'son ']

x_train = torch.tensor([
    [char_encodings[3],
     char_encodings[0],
     char_encodings[1],
     char_encodings[12]],  # hat
    [char_encodings[4],
     char_encodings[0],
     char_encodings[1],
     char_encodings[12]],  # rat
    [char_encodings[2],
     char_encodings[0],
     char_encodings[1],
     char_encodings[12]],  # cat
    [char_encodings[5],
     char_encodings[6],
     char_encodings[0],
     char_encodings[1]],  # flat
    [char_encodings[7],
     char_encodings[0],
     char_encodings[1],
     char_encodings[1]],  # matt
    [char_encodings[2],
     char_encodings[0],
     char_encodings[8],
     char_encodings[12]],  # cap
    [char_encodings[9],
     char_encodings[10],
     char_encodings[11],
     char_encodings[12]]  # son
])

y_train = torch.tensor([
    out_encodings[0],  # hat
    out_encodings[1],  # rat
    out_encodings[2],  # cat
    out_encodings[3],  # flat
    out_encodings[4],  # matt
    out_encodings[5],  # cap
    out_encodings[6]   # son
])  # emojis

print(x_train.shape)
print(y_train.shape)

hat_tensor = ('hat ', torch.tensor([[char_encodings[3], char_encodings[0], char_encodings[1], char_encodings[12]]]))
rat_tensor = ('rat ', torch.tensor([[char_encodings[4], char_encodings[0], char_encodings[1], char_encodings[12]]]))
cat_tensor = ('cat ', torch.tensor([[char_encodings[2], char_encodings[0], char_encodings[1], char_encodings[12]]]))
flat_tensor = ('flat', torch.tensor([[char_encodings[5], char_encodings[6], char_encodings[0], char_encodings[1]]]))
matt_tensor = ('matt', torch.tensor([[char_encodings[7], char_encodings[0], char_encodings[1], char_encodings[1]]]))
cap_tensor = ('cap ', torch.tensor([[char_encodings[2], char_encodings[0], char_encodings[8], char_encodings[12]]]))
son_tensor = ('son ', torch.tensor([[char_encodings[9], char_encodings[10], char_encodings[11], char_encodings[12]]]))

tensors = [hat_tensor, rat_tensor, cat_tensor, flat_tensor, matt_tensor, cap_tensor, son_tensor]

model = LongShortTermMemoryModel(encoding_size, out_encoding_size)
temp = ('hat', torch.tensor([[char_encodings[3], char_encodings[0], char_encodings[1], char_encodings[12]]]))
optimizer = torch.optim.RMSprop(model.parameters(), 0.005)
for epoch in range(52):
    if epoch % 10 == 1:
        print("epoch", epoch)
        rd.shuffle(tensors)
        for tensor in tensors:
            y = model.f(tensor[1])
            print("I=>O:", tensor[0], "=>", emojis[y.argmax(1)])
        print("\n")
        # y = model.f(torch.tensor([[char_encodings[3], char_encodings[0], char_encodings[1], char_encodings[12]]]))
        # print("hat ", emojis[y.argmax(1)])
        # y = model.f(torch.tensor([[char_encodings[4], char_encodings[0], char_encodings[1], char_encodings[12]]]))
        # print("rat ", emojis[y.argmax(1)])
        # y = model.f(torch.tensor([[char_encodings[2], char_encodings[0], char_encodings[1], char_encodings[12]]]))
        # print("cat ", emojis[y.argmax(1)])
        # y = model.f(torch.tensor([[char_encodings[5], char_encodings[6], char_encodings[0], char_encodings[1]]]))
        # print("flat", emojis[y.argmax(1)])
        # y = model.f(torch.tensor([[char_encodings[7], char_encodings[0], char_encodings[1], char_encodings[1]]]))
        # print("matt", emojis[y.argmax(1)])
        # y = model.f(torch.tensor([[char_encodings[2], char_encodings[0], char_encodings[8], char_encodings[12]]]))
        # print("cap ", emojis[y.argmax(1)])
        # y = model.f(torch.tensor([[char_encodings[9], char_encodings[10], char_encodings[11], char_encodings[12]]]))
        # print("son ", emojis[y.argmax(1)], "\n")

    model.reset()
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

# Testing


print("\n\nTASK TESTING\n\n")

rt_tensor = ('rt  ', torch.tensor([[char_encodings[4], char_encodings[1], char_encodings[12], char_encodings[12]]]))
rats_tensor = ('rats', torch.tensor([[char_encodings[4], char_encodings[0], char_encodings[1], char_encodings[9]]]))

task_tensors = {rt_tensor, rats_tensor}
for tensor in task_tensors:
    y = model.f(tensor[1])
    print("I=>O", tensor[0], "=>", emojis[y.argmax(1)])
    print(y)
    print("\n")

print("\n\nCONTROL TESTING\n\n")

for tensor in tensors:
    # print(tensor)
    y = model.f(tensor[1])
    print("I=>O:", tensor[0], "=>", emojis[y.argmax(1)])
    print(y)
    print("\n")
