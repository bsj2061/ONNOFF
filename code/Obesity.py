import torch.nn as nn
import torch.nn.functional as F
import random
import cv2
import Preprocess

class SizeModel(nn.Module):
    def __init__(self, file):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(245,64)
        self.fc2 = nn.Linear(64, 10)

        self.file = file

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)

        x = x.view(-1,256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x)

    def train(self):
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
    
    def result(self):
        results = ["하체비만", "상체비만"]
        W = 100
        H = 100
        delta = random.choice(results)
        image = preprocess.fitSize(file, W, H)
        image = preprocess.removeNoise(file)
        y_test = []
        y_pred = []
        for i in range(W*H):
            accuracy = mt.accuracy_score(y_test, y_pred)
            recall = mt.recall_score(y_test, y_pred)
            precision = mt.precision_score(y_test, y_pred)
            f1_score = mt.f1_score(y_test, y_pred)
            matrix = mt.confusion_matrix(y_test, y_pred)
            
        image, prediction = recall, delta
        dt_clf = DecisionTreeClassifier(random_state=33)
        parameters = {'max_depth': [3, 5, 7],
                      'min_samples_split': [3, 5],
                      'splitter': ['best', 'random']}

        grid_dt = GridSearchCV(dt_clf, # estimator 객체,
                              param_grid = parameters, cv = 5))

        grid_dt.fit(X_train, y_train)

        result = pd.DataFrame(grid_dt.cv_results_['params'])
         result['mean_test_score'] = grid_dt.cv_results_['mean_test_score']

        return prediction
