[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=7811586&assignment_repo_type=AssignmentRepo)
# Лабораторная работа по курсу "Искусственный интеллект"
# Классификация изображений.

| Студент | *ФИО* |
|------|------|
| Группа  | *№* |
| Оценка 1 (обучение "с нуля") | *X* |
| Оценка 2 (transfer learning) | *X* |
| Проверил | Сошников Д.В. |

> *Комментарии проверяющего*
### Задание

Решить задачу классификации пород кошек и собак на основе датасета [Oxford-IIIT](https://www.robots.ox.ac.uk/~vgg/data/pets/).

![Dataset we will deal with](images/data.png)

#### Задание 1: Классификация Pet Faces

Обучить свёрточную нейронную сеть для классификации пород кошек и собак на основе упрощённого датасета **Pet Faces**. Самостоятельно придумать архитектуру сети, реализовать предобработку входных данных.

Для загрузки датасета используйте следующий код:

```python
!wget https://mslearntensorflowlp.blob.core.windows.net/data/petfaces.tar.gz
!tar xfz petfaces.tar.gz
!rm petfaces.tar.gz
```

В качестве результата необходимо:

* Посчитать точность классификатора на тестовом датасете
* Посчитать точность двоичной классификации "кошки против собак" на текстовом датасете
* Построить confusion matrix
* **[На хорошую и отличную оценку]** Посчитайте top-3 accuracy
* **[На отличную оценку]** Выполнить оптимизацию гиперпараметров: архитектуры сети, learning rate, количества нейронов и размеров фильтров.

Решение оформите в файле [Faces.ipynb](Faces.ipynb).

Использовать нейросетевой фреймворк в соответствии с вариантом задания:
   * Чётные варианты - PyTorch
   * Нечётные варианты - Tensorflow/Keras
#### Задание 2: Классификация полных изображений с помощью transfer learning

Используйте оригинальный датасет **Oxford Pets** и предобученные сети VGG-16/VGG-19 и ResNet для построение классификатора пород. Для загрузки датасета используйте код:

```python
!wget https://mslearntensorflowlp.blob.core.windows.net/data/oxpets_images.tar.gz
!tar xfz oxpets_images.tar.gz
!rm oxpets_images.tar.gz
```

В качестве результата необходимо:

* Посчитать точность классификатора на тестовом датасете отдельно для VGG-16/19 и ResNet, для дальнейших действий выбрать сеть с лучшей точностью
* Посчитать точность двоичной классификации "кошки против собак" на текстовом датасете
* Построить confusion matrix
* **[На отличную оценку]** Посчитайте top-3 и top-5 accuracy

Решение оформите в файле [Pets.ipynb](Pets.ipynb).

Использовать нейросетевой фреймворк, отличный от использованного в предыдущем задании, в соответствии с вариантом задания:
   * Нечётные варианты - PyTorch
   * Чётные варианты - Tensorflow/Keras

## Отчет

#### Задание 1: Классификация Pet Faces
Подготавливаем данные для многоклассовой класификации:

    def train_test_dataset(dataset, val_split=0.25):
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, random_state=42)
        datasets = {}
        datasets['train'] = Subset(dataset, train_idx)
        datasets['test'] = Subset(dataset, val_idx)
        return datasets

    dataset = ImageFolder('petfaces', transform=transforms.Compose([transforms.Resize((128,128)),
                                                         transforms.ToTensor(),
                                                          transforms.Normalize([0.5, 0.5, 0.5],
                                                                                [0.2, 0.2, 0.2])]))
              
    datasets = train_test_dataset(dataset)
    
    train_loader = DataLoader(dataset=datasets['train'], batch_size=20,shuffle=True) 
    test_loader = DataLoader(dataset=datasets['test'], batch_size=20, shuffle=False)
    
Подготавливаем данные для двоичной класификации:

    os.makedirs('./output')
    os.makedirs('./output/cats')
    os.makedirs('./output/dogs')
    petfaces = os.listdir('petfaces')

    for i, im in enumerate(petfaces):
      if 'cat' in petfaces[i]:
        cats = os.listdir(os.path.join('petfaces',petfaces[i]))
        for n, cat in enumerate(cats):
          shutil.copy(os.path.join(os.path.join('petfaces',petfaces[i]), cats[n]), './output/cats')
      if 'dog' in petfaces[i]:
        if len(os.listdir('./output/cats')) > len(os.listdir('./output/dogs')):
          dogs = os.listdir(os.path.join('petfaces',petfaces[i]))
          for m, dog in enumerate(dogs):
            shutil.copy(os.path.join(os.path.join('petfaces',petfaces[i]), dogs[m]), './output/dogs')

    dataset1 = ImageFolder('output', transform=transforms.Compose([transforms.Resize((128,128)),
                                                         transforms.ToTensor(),
                                                          transforms.Normalize([0.5, 0.5, 0.5],
                                                                                [0.2, 0.2, 0.2])]))

    datasets1 = train_test_dataset(dataset1)
    train_loader1 = DataLoader(dataset=datasets1['train'], batch_size=20,shuffle=True) 
    test_loader1 = DataLoader(dataset=datasets1['test'], batch_size=20, shuffle=False)
                                                                            
Архитектура сети:

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 8, 3)
      #      self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(8, 8, 3)
            self.pool = nn.MaxPool2d(4, 4)
            #self.conv3 = nn.Conv2d(32, 64, 7)
            #self.pool = nn.MaxPool2d(2, 2)
            #self.conv4 = nn.Conv2d(64, 64, 7)
            #self.pool = nn.MaxPool2d(2, 2)
            self.fl = nn.Flatten()
            self.fc1 = nn.Linear(7688, 512)
         #   self.fc2 = nn.Linear(1000, 350)
            self.fc2 = nn.Linear(512, 35)
        def forward(self, x):
            #x = self.pool(F.softsign(self.conv1(x)))
            #x = self.pool(F.softsign(self.conv2(x)))
            x = self.conv1(x)
            x = self.pool(F.softsign(self.conv2(x)))
            #x = self.pool(F.relu(self.conv3(x)))
            #x = self.pool(F.relu(self.conv4(x)))
            x = self.fl(x)
            x = F.softsign(self.fc1(x))
            #x = self.fl(x)
            #x = F.softsign(self.fc2(x))
            x = self.fc2(x)
            return x

Обучение

    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(5):  
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print('%d loss: %.3f' %
                      (epoch + 1, loss.item()))
    print('Finished Training')
    
    Вывод:
    1 loss: 2.200
    2 loss: 1.824
    3 loss: 0.384
    4 loss: 0.550
    5 loss: 0.047
    Finished Training
    
Метрики

точность на тренировочной выборке:
Accuracy 99.95847176079734%
точность на тестовой выборке:
Accuracy 55.04358655043587%
точность по классам:
[0.0,
 0.5504358655043586,
 0.688667496886675,
 0.7708592777085927,
 0.8119551681195517,
 0.8455790784557908,
 0.8729763387297634,
 0.8941469489414695,
 0.9103362391033624,
 0.9165628891656289,
 0.925280199252802,
 0.933997509339975,
 0.9389788293897883,
 0.9476961394769614,
 0.9514321295143213,
 0.9539227895392279,
 0.9613947696139477,
 0.9651307596513076,
 0.9701120797011208,
 0.9713574097135741,
 0.9738480697384807,
 0.9763387297633873,
 0.9800747198007472,
 0.9838107098381071,
 0.987546699875467,
 0.9887920298879203,
 0.9900373599003736,
 0.9925280199252802,
 0.9937733499377335,
 0.9950186799501868,
 0.9975093399750934,
 0.9975093399750934,
 0.9987546699875467,
 1.0,
 1.0]
 
 Архитектура сети для двуклассовой класификации:
 
     class Net1(nn.Module):
        def __init__(self):
            super(Net1, self).__init__()
            self.conv1 = nn.Conv2d(3, 12,8)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(12, 32,8)
            self.pool = nn.MaxPool2d(2, 2)
            #self.conv3 = nn.Conv2d(32, 64, 7)
            #self.pool = nn.MaxPool2d(2, 2)
            #self.conv4 = nn.Conv2d(64, 64, 7)
            #self.pool = nn.MaxPool2d(2, 2)
            self.fl = nn.Flatten()
            self.fc1 = nn.Linear(21632, 2000)
         #   self.fc2 = nn.Linear(1000, 350)
            self.fc2 = nn.Linear(2000, 2)
        def forward(self, x):
           # x = self.conv1(x)
            x = self.pool(F.softsign(self.conv1(x)))
            x = self.pool(F.softsign(self.conv2(x)))
            #x = self.pool(F.relu(self.conv3(x)))
            #x = self.pool(F.relu(self.conv4(x)))
            x = self.fl(x)
            x = F.softsign(self.fc1(x))
            #x = self.fl(x)
            #x = F.softsign(self.fc2(x))
            x = self.fc2(x)
            return x
            
 Обучение
 
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net1.parameters(), lr=0.001)
    for epoch in range(5):  
        for i, data in enumerate(train_loader1):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net1(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print('%d loss: %.3f' %
                      (epoch + 1, loss.item()))
    print('Finished Training')
    
    Вывод
    
    1 loss: 0.583
    2 loss: 0.432
    3 loss: 0.379
    4 loss: 0.231
    5 loss: 0.251
    Finished Training
    
точность на тренировочной выборке:
Accuracy 88.66930171277997%
точность на тестовой выборке:
Accuracy 82.80632411067194%
confusion_matrix:
array([[764,  68],
       [104, 582]])
       
#### Задание 2: Классификация полных изображений с помощью transfer learning




## Codespaces

По возможности, используйте GitHub Codespaces для выполнения работы. По результатам, дайте обратную связь:
1. Что понравилось?
1. Что не понравилось?
1. Какие ошибки или существенные затруднения в работе вы встречали? (По возможности, будьте как можно более подробны, указывайте шаги для воспроизведения ошибок)

## Материалы для изучения

* [Deep Learning for Image Classification Workshop](https://github.com/microsoft/workshop-library/blob/main/full/deep-learning-computer-vision/README.md)
* [Convolutional Networks](https://github.com/microsoft/AI-For-Beginners/blob/main/4-ComputerVision/07-ConvNets/README.md)
* [Transfer Learning](https://github.com/microsoft/AI-For-Beginners/blob/main/4-ComputerVision/08-TransferLearning/README.md)
