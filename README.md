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

В начале я разделила картинки в папки по классам(проды для многоклассовой класификации и кошки и собаки для двуклассовой), а затем создала папки для тренировочных и тестовых выборок (также для многокласовой и двукласовой класификации)

Затем: 

Для VGG19

    data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
    train = data_gen.flow_from_directory('./output1/train', target_size=img_size, shuffle=False, batch_size=20)
    test = data_gen.flow_from_directory('./output1/test', target_size=img_size, shuffle=False, batch_size=20)
    train_features = vgg.predict_generator(train,steps=5542//20)
    test_features = vgg.predict_generator(test,steps=1848//20)
    
Сеть и обучение:

    model = keras.models.Sequential()
    model.add(Flatten(input_shape=train_features.shape[1:]))
    model.add(Dense(512, activation='softsign'))
    model.add(Dropout(0.5))
    model.add(Dense(37, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_features, train_labels,
              epochs=25,
              batch_size=20,
              validation_data=(test_features, test_labels))
 Вывод:
 
 Epoch 1/25
277/277 [==============================] - 12s 43ms/step - loss: 3.6639 - accuracy: 0.1014 - val_loss: 2.7551 - val_accuracy: 0.2245
Epoch 2/25
277/277 [==============================] - 12s 42ms/step - loss: 2.6206 - accuracy: 0.2670 - val_loss: 2.3015 - val_accuracy: 0.3538
Epoch 3/25
277/277 [==============================] - 12s 43ms/step - loss: 2.1675 - accuracy: 0.3742 - val_loss: 2.3229 - val_accuracy: 0.3359
Epoch 4/25
277/277 [==============================] - 12s 43ms/step - loss: 1.8908 - accuracy: 0.4478 - val_loss: 2.2377 - val_accuracy: 0.3598
Epoch 5/25
277/277 [==============================] - 12s 43ms/step - loss: 1.6602 - accuracy: 0.5171 - val_loss: 2.1294 - val_accuracy: 0.3853
Epoch 6/25
277/277 [==============================] - 12s 43ms/step - loss: 1.4866 - accuracy: 0.5547 - val_loss: 2.2608 - val_accuracy: 0.3658
Epoch 7/25
277/277 [==============================] - 12s 43ms/step - loss: 1.3312 - accuracy: 0.5982 - val_loss: 2.1620 - val_accuracy: 0.3951
Epoch 8/25
277/277 [==============================] - 12s 43ms/step - loss: 1.1833 - accuracy: 0.6473 - val_loss: 2.1997 - val_accuracy: 0.3793
Epoch 9/25
277/277 [==============================] - 12s 43ms/step - loss: 1.0600 - accuracy: 0.6821 - val_loss: 2.1970 - val_accuracy: 0.3962
Epoch 10/25
277/277 [==============================] - 12s 43ms/step - loss: 0.9508 - accuracy: 0.7128 - val_loss: 2.1244 - val_accuracy: 0.4207
Epoch 11/25
277/277 [==============================] - 12s 43ms/step - loss: 0.8384 - accuracy: 0.7446 - val_loss: 2.2000 - val_accuracy: 0.3957
Epoch 12/25
277/277 [==============================] - 12s 43ms/step - loss: 0.7482 - accuracy: 0.7688 - val_loss: 2.4102 - val_accuracy: 0.3777
Epoch 13/25
277/277 [==============================] - 12s 43ms/step - loss: 0.6796 - accuracy: 0.7908 - val_loss: 2.4685 - val_accuracy: 0.3886
Epoch 14/25
277/277 [==============================] - 12s 44ms/step - loss: 0.6075 - accuracy: 0.8168 - val_loss: 2.3389 - val_accuracy: 0.3995
Epoch 15/25
277/277 [==============================] - 12s 44ms/step - loss: 0.5359 - accuracy: 0.8359 - val_loss: 2.3739 - val_accuracy: 0.4125
Epoch 16/25
277/277 [==============================] - 12s 43ms/step - loss: 0.4610 - accuracy: 0.8556 - val_loss: 2.3375 - val_accuracy: 0.4033
Epoch 17/25
277/277 [==============================] - 12s 43ms/step - loss: 0.4277 - accuracy: 0.8713 - val_loss: 2.4129 - val_accuracy: 0.4130
Epoch 18/25
277/277 [==============================] - 12s 43ms/step - loss: 0.3934 - accuracy: 0.8821 - val_loss: 2.4889 - val_accuracy: 0.4027
Epoch 19/25
277/277 [==============================] - 12s 44ms/step - loss: 0.3441 - accuracy: 0.8931 - val_loss: 2.4370 - val_accuracy: 0.4114
Epoch 20/25
277/277 [==============================] - 12s 44ms/step - loss: 0.3165 - accuracy: 0.9016 - val_loss: 2.5910 - val_accuracy: 0.4125
Epoch 21/25
277/277 [==============================] - 12s 44ms/step - loss: 0.2922 - accuracy: 0.9083 - val_loss: 2.6662 - val_accuracy: 0.4082
Epoch 22/25
277/277 [==============================] - 12s 43ms/step - loss: 0.2619 - accuracy: 0.9208 - val_loss: 2.6326 - val_accuracy: 0.4168
Epoch 23/25
277/277 [==============================] - 12s 43ms/step - loss: 0.2319 - accuracy: 0.9265 - val_loss: 2.6330 - val_accuracy: 0.4179
Epoch 24/25
277/277 [==============================] - 12s 44ms/step - loss: 0.2139 - accuracy: 0.9319 - val_loss: 2.8003 - val_accuracy: 0.4098
Epoch 25/25
277/277 [==============================] - 12s 43ms/step - loss: 0.1951 - accuracy: 0.9433 - val_loss: 2.7575 - val_accuracy: 0.4005
<keras.callbacks.History at 0x7f0148ad2610>

confusion_matrix:

[18  2  0  4  2  1  3  0  3  1  0  2  0  0  1  0  0  5  3  1  0  1  0  0
  1  0  0  0  0  0  0  0  0  1  0  0  1]

[ 3 21  0  1  2  4  4  0  0  1  0  1  0  0  1  0  0  2  1  0  0  2  0  0
  0  0  0  3  0  0  0  0  0  1  0  0  3]

[ 0  1 12  0  2  0  2  3  7  0  0  0  0  0  0  0  0  1  2  0  0  1  2  4
  3  2  0  1  1  0  0  1  1  1  2  0  1]

[ 2  0  1 33  2  0  3  0  0  2  0  1  0  0  0  0  0  2  0  0  0  0  0  0
  0  1  1  1  0  1  0  0  0  0  0  0  0]

[ 0  3  0  2 19  2  3  0  1 12  0  0  0  0  0  0  1  1  0  0  0  0  0  0
  0  0  1  1  0  1  0  0  1  0  2  0  0]

[ 1  3  0  0  1 35  1  0  0  1  0  0  0  1  0  0  1  2  1  0  0  0  0  0
  0  0  0  1  0  1  0  0  0  1  0  0  0]

[ 0  1  0  0  2  1 25  4  3  1  1  0  0  0  0  0  0  1  4  0  0  0  0  0
  0  0  1  1  1  0  0  2  2  0  0  0  0]

[ 0  0  0  1  1  0  9 22  2  0  0  0  0  0  0  0  1  1  3  0  0  1  1  1
  0  2  0  1  1  1  0  0  0  1  0  0  1]

[ 1  0  3  0  1  0  5  4 19  1  1  0  0  0  0  0  2  1  1  0  0  3  1  1
  0  0  0  0  3  1  0  0  0  0  0  1  1]

[ 5  1  2  1 10  0  1  0  1 16  1  2  0  1  0  0  1  1  0  0  0  0  0  0
  0  2  0  2  0  0  0  0  3  0  0  0  0]

[ 2  3  2  2  2  0  0  0  2  2 21  1  0  0  0  0  0  4  0  0  0  1  1  1
  0  1  3  0  0  0  0  0  0  1  0  0  1]

[ 0  1  1  1  2  0  0  0  1  1  1 20  2  0  2  0  0 11  0  0  1  1  0  0
  0  0  1  0  0  0  0  0  1  1  0  0  2]

[ 2  0  0  0  0  0  0  0  0  1  1  0 10  3  0  1  4  6  3  2  0  6  1  2
  0  1  0  0  0  0  3  2  0  1  0  1  0]

[0 2 0 0 0 0 1 0 0 0 0 0 3 8 0 1 4 3 3 1 5 4 1 1 0 2 1 3 0 0 1 0 1 3 2 0 0]

[ 0  0  0  0  0  0  0  0  1  0  0  0  0  2 22  5  4  2  9  1  2  0  0  0
  0  1  0  0  0  1  0  0  0  1  0  0  0]

[ 0  0  0  0  0  0  0  0  0  0  1  0  1  4  5 10  4  5  5  2  0  1  0  1
  0  1  2  0  0  0  7  0  1  0  0  0  0]

[ 1  2  0  0  0  1  0  0  0  1  0  0  1  4  4  4 18  1  1  1  0  0  1  0
  1  1  2  3  0  2  0  0  0  0  0  0  1]

[ 1  1  0  0  0  1  1  0  0  0  1  2  1  1  0  0  2 23  2  0  1  0  0  0
  1  0  3  1  1  0  1  0  2  2  0  0  2]

[ 0  0  0  0  0  0  0  0  1  0  0  0  1  0  0  0  1  0 25  2  1  1  0  2
  0  4  1  4  1  0  0  0  0  0  1  0  3]

[ 0  0  0  0  3  0  0  0  0  0  0  1  0  0  1  0  4  0 16 14  0  6  1  1
  0  0  0  0  0  0  1  0  0  0  0  1  1]

[ 1  0  0  0  1  0  4  0  0  0  0  0  0  4  0  0  6  1  5  4 12  0  1  1
  0  0  0  1  1  1  1  0  2  0  4  0  0]

[ 0  0  1  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  1  2  0 34  2  1
  0  0  0  2  2  0  2  2  0  0  0  0  0]

[ 0  1  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0 11  1  0  5 17  0
  2  3  0  3  1  0  2  0  0  0  0  2  1]

[ 0  1  0  1  0  0  0  0  1  0  0  0  0  0  1  3  1  1  1  0  0  5  3 22
  2  0  0  0  1  0  2  0  2  0  1  1  1]

[ 0  0  0  1  1  0  1  0  0  0  0  0  0  0  0  0  1  0  5  0  0  0  2  3
 23  5  0  3  0  1  1  0  2  0  0  0  1]

[ 1  0  0  1  0  0  0  0  0  1  0  0  0  0  0  0  2  0  7  3  0  0  2  1
  2 25  1  3  0  0  0  0  0  0  0  1  0]

[ 2  1  0  0  1  0  0  0  0  0  0  0  0  2  1  1  7  8  1  1  1  0  1  0
  0  3 10  1  0  1  0  0  1  2  1  0  4]

[ 0  1  0  0  1  0  0  0  0  0  0  1  0  0  0  0  0  0  4  1  1  0  0  0
  1  2  0 34  1  0  0  0  1  0  1  1  0]

[ 0  0  0  0  0  0  0  6  6  0  0  0  0  0  0  0  0  1  2  0  0  2  3  1
  1  0  0  1 22  0  0  1  1  1  0  0  2]

[ 0  0  1  0  1  2  0  0  0  0  0  0  1  1  0  1  7  2  1  0  0  0  0  0
  0  2  1  1  1 25  1  0  0  0  1  0  1]

[ 0  0  1  0  0  0  0  0  0  0  0  0  2  3  2  1  1  2  6  0  0  1  1  5
  0  1  0  0  1  0 21  0  1  0  0  0  1]

[ 1  0  3  0  1  0  0  2  3  0  0  1  0  0  1  0  2  0  4  0  0  9  3  0
  0  0  0  0  1  0  0 16  1  1  0  1  0]

[ 1  0  1  0  0  0  2  0  0  3  1  1  0  0  1  0  0  1  1  1  0  1  0  0
  1  0  0  6  0  0  0  1 23  1  1  0  3]

[ 0  2  1  0  0  0  2  0  1  0  1  0  1  1  0  1  2  5  3  1  0  3  1  0
  0  3  0  0  2  0  2  0  0 15  1  0  2]

[ 0  0  0  1  1  0  0  0  1  0  1  0  1 10  0  0  3  0  5  1  1  0  0  1
  0  2  2  4  0  3  1  0  0  1 11  0  0]

[ 0  1  1  0  0  3  1  2  0  0  0  0  1  0  0  0  2  0  8  1  0  6  3  1
  0  0  1  2  0  0  0  0  1  0  0 12  4]

[ 0  0  0  0  1  0  1  0  0  0  0  1  0  0  0  1  1  0  3  1  1  0  2  0
  0  0  0  1  1  0  0  0  2  1  0  0 24]
 
top_k_accuracy:

    top_k_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)

    top_k_acc.__name__ = 'top_k_acc'

    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy','top_k_categorical_accuracy',top_k_acc])

    model.evaluate(test_features, test_labels)
    
 58/58 [==============================] - 1s 10ms/step - loss: 2.7575 - accuracy: 0.4005 - top_k_categorical_accuracy: 0.1625 - top_k_acc: 0.0891
[2.757471799850464,
 0.4005434811115265,
 0.16249999403953552,
 0.08913043141365051]
 
 для 5:
 
 58/58 [==============================] - 1s 10ms/step - loss: 2.7575 - accuracy: 0.4005 - top_k_categorical_accuracy: 0.1625 - top_k_acc: 0.1625
[2.757471799850464,
 0.4005434811115265,
 0.16249999403953552,
 0.16249999403953552]
 
 Для ReNet
 
    train_features1 = rn.predict_generator(train,steps=5542//20)
    test_features1 = rn.predict_generator(test,steps=1848//20)
    
 Сеть и обучение:
 
    model = keras.models.Sequential()
    model.add(Flatten(input_shape=train_features1.shape[1:]))
    model.add(Dense(512, activation='softsign'))
    model.add(Dropout(0.5))
    model.add(Dense(37, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_features1, train_labels1,
              epochs=5,
              batch_size=20,
              validation_data=(test_features1, test_labels1))
              
 Epoch 1/5
277/277 [==============================] - 44s 159ms/step - loss: 4.2733 - accuracy: 0.0332 - val_loss: 3.8166 - val_accuracy: 0.0397
Epoch 2/5
277/277 [==============================] - 43s 156ms/step - loss: 4.0392 - accuracy: 0.0394 - val_loss: 3.7273 - val_accuracy: 0.0440
Epoch 3/5
277/277 [==============================] - 43s 156ms/step - loss: 3.9055 - accuracy: 0.0514 - val_loss: 3.5428 - val_accuracy: 0.0701
Epoch 4/5
277/277 [==============================] - 43s 156ms/step - loss: 3.7779 - accuracy: 0.0590 - val_loss: 3.5562 - val_accuracy: 0.0560
Epoch 5/5
277/277 [==============================] - 43s 155ms/step - loss: 3.6813 - accuracy: 0.0673 - val_loss: 3.4121 - val_accuracy: 0.0750
<keras.callbacks.History at 0x7f0143e252d0>

confusion_matrix:

[ 0  0  1  0  0  9  0  0  0  2  0  0  0  0  0  0  0  3  0  0  0  0  0  0
  6  0  8  0  3  0  0  0  0  0  0 18  0]

[ 0  0  2  0  0  4  0  0  0  4  0  0  0  0  1  2  0  4  0  0  0  0  0  0
  7  0 11  0  1  0  0  0  0  0  0 14  0]

[ 0  0  3  0  0  0  0  0  0  5  0  0  0  0  0  0  0  8  0  0  0  0  0  1
 14  0  0  0  1  0  0  0  0  0  0 18  0]

[ 0  0  4  3  0  7  0  0  0 12  0  0  0  0  0  0  0  3  0  0  0  0  0  0
  6  0  4  0  1  0  0  0  0  0  0 10  0]

[ 0  0  4  0  0  7  0  0  0  9  0  0  0  0  0  0  0  3  0  0  0  0  0  0
  9  0  1  0  2  0  0  0  0  0  0 15  0]

[ 0  0  1  1  0 12  0  0  0  4  0  0  0  0  0  3  0  2  0  0  0  0  0  0
  3  0  8  0  0  0  0  0  0  0  0 16  0]

[ 0  0  2  0  0  3  1  0  0  7  0  0  0  0  0  0  0  3  0  0  0  0  0  1
 11  0  0  0  1  0  0  0  0  0  0 21  0]

[ 0  0  1  0  0  1  0  0  0  4  0  0  0  0  0  0  0  0  0  0  0  0  0  1
  7  0  0  0  6  0  0  0  0  0  0 30  0]

[ 0  0  5  0  0  4  0  0  0  5  0  0  0  0  0  0  0  0  0  0  0  0  0  3
 10  0  1  0  3  0  0  0  0  0  0 19  0]

[ 0  0  2  0  0 10  0  0  0  5  0  0  0  0  0  0  0  4  0  0  0  0  0  0
 10  0  4  0  1  0  0  0  0  0  0 14  0]

[ 0  0  4  1  0  6  0  0  0  1  3  0  1  0  0  0  0 10  0  0  0  0  0  1
 13  0  1  0  0  0  0  0  0  0  0  9  0]

[ 0  0  0  0  0 14  0  0  0  1  1  0  0  0  0  1  0 13  0  0  0  0  0  1
  4  0  7  0  0  0  1  0  0  0  0  7  0]

[ 0  0  1  0  0  2  0  0  0  1  0  0  3  0  0  5  0 12  0  0  0  1  0  3
  2  0  6  0  0  1  2  0  0  0  0 11  0]

[ 0  0  0  0  0  2  0  0  0  3  0  0  1  0  0  2  0  9  0  0  0  0  0  1
 12  0  5  0  1  0  1  0  0  0  0 13  0]

[ 0  0  0  0  0  6  0  0  0  1  0  0  0  0  4  2  0 11  0  0  0  0  0  2
  7  0  9  0  0  0  3  0  0  0  0  6  0]

[ 0  0  0  0  0  2  0  0  0  0  0  0  1  0  0  6  0 13  0  0  0  0  0  5
  2  0  7  1  0  0  0  0  0  0  0 13  0]

[ 0  0  0  0  0  3  0  0  0  3  0  0  1  0  2  4  0  4  0  0  0  0  0  2
 10  0  7  0  0  0  3  0  0  0  0 11  0]

[ 0  0  4  0  0  2  0  0  0  2  0  0  0  0  1  2  0 12  0  0  0  0  0  1
  3  0 10  0  0  0  2  0  0  0  1 10  0]

[ 0  0  0  0  0  0  0  0  0  2  0  0  0  0  0  2  0  4  0  0  0  0  0  3
 10  0  8  0  0  0  0  0  0  0  1 18  0]

[ 0  0  0  0  0  2  0  0  0  1  0  0  1  0  0  1  0  3  0  0  0  0  0  3
 11  0  7  0  0  0  1  0  0  0  2 18  0]

[ 0  0  0  0  0  1  0  0  0  1  0  0  1  0  0  4  0  2  0  0  1  0  0  0
  6  0 15  0  0  0  1  0  0  0  0 18  0]

[ 0  0  1  0  0  3  0  0  0  3  0  0  1  0  0  0  0  5  0  0  0  0  0  1
 11  0  3  0  0  0  0  0  0  0  0 22  0]

[ 0  0  0  0  0  1  0  0  0  2  0  0  0  0  0  0  0  6  0  0  0  0  0  2
 15  0  4  1  0  0  0  0  0  0  0 19  0]

[ 0  0  1  0  0  2  0  0  0  0  0  0  0  0  0  1  0 12  0  0  0  0  0 10
  6  0  4  0  0  0  1  0  0  0  0 13  0]

[ 0  0  2  0  0  1  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  1
 28  0  3  1  0  0  0  0  0  0  0 13  0]

[ 0  0  0  0  0  2  0  0  0  5  0  0  0  0  0  0  0  1  0  0  0  0  0  2
 21  0  3  0  1  0  0  0  0  0  0 15  0]

[ 0  0  2  0  0 10  0  1  0  0  1  0  1  0  0  1  0  7  0  0  0  0  0  1
  7  0 10  0  1  0  0  0  0  0  0  8  0]

[ 0  0  1  0  0  2  0  1  0  5  0  0  0  0  0  0  0  2  0  0  0  0  0  1
 18  0  5  3  0  0  0  0  0  0  0 12  0]

[ 0  0  0  0  0  1  0  0  0  1  0  0  0  0  0  0  0  2  0  0  0  0  0  1
  8  0  0  0  3  0  0  0  0  0  0 34  0]

[ 0  0  1  0  0  6  0  0  0  3  0  0  3  0  0  1  0 12  0  0  0  0  0  0
  9  0  5  0  0  0  0  0  0  0  0 10  0]

[ 0  0  0  0  0  2  0  0  0  0  0  0  1  0  0  2  0 11  0  0  0  0  0 14
  3  0  2  0  0  0  8  0  0  0  1  6  0]

[ 0  0  3  0  0  1  0  0  0  1  0  1  0  0  0  0  0  4  0  0  0  0  0  2
  7  0  4  0  0  0  0  0  0  0  1 26  0]

[ 0  0  3  0  0  4  0  1  0  6  0  0  0  0  0  0  0  6  0  0  0  0  0  2
 14  0  5  0  2  0  1  0  0  0  0  6  0]

[ 0  0  0  0  0  3  0  0  0  2  0  0  0  0  0  0  0  6  0  0  0  0  0  0
  6  0  6  0  1  0  0  0  0  0  1 25  0]

[ 0  0  0  0  0  2  0  0  0  2  1  0  1  0  0  4  0  7  0  0  0  0  0  3
  2  0 11  0  0  0  5  0  0  0  1 11  0]

[ 0  0  0  0  0  3  0  0  0  1  0  0  0  0  0  1  0  1  0  0  0  1  0  1
 18  0  2  0  0  0  0  0  0  0  0 22  0]

[ 0  0  0  0  0  2  1  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0
 20  0  3  0  0  0  0  0  0  0  0 14  0]
 
 top_k_accuracy:
 
 58/58 [==============================] - 2s 32ms/step - loss: 3.4121 - accuracy: 0.0750 - top_k_categorical_accuracy: 0.0016 - top_k_acc: 5.4348e-04
[3.4121437072753906,
 0.07500000298023224,
 0.0016304347664117813,
 0.0005434782360680401]
 
 58/58 [==============================] - 2s 32ms/step - loss: 3.4121 - accuracy: 0.0750 - top_k_categorical_accuracy: 0.0016 - top_k_acc: 0.0016
[3.4121437072753906,
 0.07500000298023224,
 0.0016304347664117813,
 0.0016304347664117813]
 
 Так как точность на VGG лучше чем на ReNet для двоичной класификации используем VGG
 
    data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
    train = data_gen.flow_from_directory('./output3/train', target_size=img_size, shuffle=False, batch_size=20, class_mode='binary')
    test = data_gen.flow_from_directory('./output3/test', target_size=img_size, shuffle=False, batch_size=20, class_mode='binary')
    train_features2 = vgg.predict_generator(train,steps=5542//20)
    test_features2 = vgg.predict_generator(test,steps=1848//20)
    
 Сеть о оббучение:
 
    model = keras.models.Sequential()
    model.add(Flatten(input_shape=train_features2.shape[1:]))
    model.add(Dense(512, activation='softsign'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_features2, train_labels2,
              epochs=15,
              batch_size=20,
              validation_data=(test_features2, test_labels2))
              
Epoch 1/15
277/277 [==============================] - 13s 44ms/step - loss: 0.7529 - accuracy: 0.6338 - val_loss: 0.6541 - val_accuracy: 0.6788
Epoch 2/15
277/277 [==============================] - 12s 44ms/step - loss: 0.6504 - accuracy: 0.6673 - val_loss: 0.5784 - val_accuracy: 0.7016
Epoch 3/15
277/277 [==============================] - 12s 44ms/step - loss: 0.5876 - accuracy: 0.7027 - val_loss: 0.6132 - val_accuracy: 0.6957
Epoch 4/15
277/277 [==============================] - 12s 44ms/step - loss: 0.5449 - accuracy: 0.7247 - val_loss: 0.5916 - val_accuracy: 0.7076
Epoch 5/15
277/277 [==============================] - 12s 44ms/step - loss: 0.5160 - accuracy: 0.7482 - val_loss: 0.5951 - val_accuracy: 0.6989
Epoch 6/15
277/277 [==============================] - 12s 44ms/step - loss: 0.5013 - accuracy: 0.7482 - val_loss: 0.5643 - val_accuracy: 0.7277
Epoch 7/15
277/277 [==============================] - 12s 44ms/step - loss: 0.4873 - accuracy: 0.7648 - val_loss: 0.5845 - val_accuracy: 0.7196
Epoch 8/15
277/277 [==============================] - 12s 44ms/step - loss: 0.4738 - accuracy: 0.7718 - val_loss: 0.5813 - val_accuracy: 0.7250
Epoch 9/15
277/277 [==============================] - 12s 44ms/step - loss: 0.4653 - accuracy: 0.7742 - val_loss: 0.6169 - val_accuracy: 0.7234
Epoch 10/15
277/277 [==============================] - 12s 44ms/step - loss: 0.4544 - accuracy: 0.7819 - val_loss: 0.6033 - val_accuracy: 0.7364
Epoch 11/15
277/277 [==============================] - 12s 44ms/step - loss: 0.4452 - accuracy: 0.7865 - val_loss: 0.5987 - val_accuracy: 0.7283
Epoch 12/15
277/277 [==============================] - 12s 44ms/step - loss: 0.4348 - accuracy: 0.7986 - val_loss: 0.6695 - val_accuracy: 0.7239
Epoch 13/15
277/277 [==============================] - 12s 44ms/step - loss: 0.4343 - accuracy: 0.7984 - val_loss: 0.6231 - val_accuracy: 0.6962
Epoch 14/15
277/277 [==============================] - 12s 44ms/step - loss: 0.4298 - accuracy: 0.8020 - val_loss: 0.6422 - val_accuracy: 0.6685
Epoch 15/15
277/277 [==============================] - 12s 44ms/step - loss: 0.4145 - accuracy: 0.8101 - val_loss: 0.7325 - val_accuracy: 0.7223
<keras.callbacks.History at 0x7f01442febd0>

confusion_matrix:

array([[1248,    0],
       [ 592,    0]])



## Codespaces

По возможности, используйте GitHub Codespaces для выполнения работы. По результатам, дайте обратную связь:
1. Что понравилось?
1. Что не понравилось?
1. Какие ошибки или существенные затруднения в работе вы встречали? (По возможности, будьте как можно более подробны, указывайте шаги для воспроизведения ошибок)

## Материалы для изучения

* [Deep Learning for Image Classification Workshop](https://github.com/microsoft/workshop-library/blob/main/full/deep-learning-computer-vision/README.md)
* [Convolutional Networks](https://github.com/microsoft/AI-For-Beginners/blob/main/4-ComputerVision/07-ConvNets/README.md)
* [Transfer Learning](https://github.com/microsoft/AI-For-Beginners/blob/main/4-ComputerVision/08-TransferLearning/README.md)
