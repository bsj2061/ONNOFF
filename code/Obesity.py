import torch.nn as nn
import torch.nn.functional as F
import random

class SizeModel(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(245,64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)

        x = x.view(-1,256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x)


model = SizeModel()
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    tot_loss = 0

    for train_x, train_y in train_loader:
        train_x, train_y = Variable(train_x), Variable(train_y)
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.data.item()

    if (epoch+1) % 10 == 0:
        print(epoch+1, totla_loss)

test_x, test_y = Variable(test_x), Variable(test_Y)
result = torxh.max(model(test_x).data, 1)[1]
accuracy = sum(test_y.cpu().data.numpy() == result.cpu().numpy()) / len(test_y.cpu().data.numpy())
accuracy
