import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import sampler
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# This was created for when using a planner with the network
def create_image(partial_info_binary_matrices, path_matrix, final_actions_binary_matrices):
    image = list()
    
    for i in range(len(partial_info_binary_matrices)):
        
        for partial_info in partial_info_binary_matrices[i]:
            image.append(partial_info)

        image.append(path_matrix)

        for action in final_actions_binary_matrices[i]:
            image.append(action)

        # this is needed, because torch complains otherwise that converting a list is too slow
        # it's better to use a np array because of the way a np array is stored in memory (contiguous)
        image = np.array(image)
        
    return torch.IntTensor(image)


class PlanningDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        # The label is converted to a float and in a list because the NN will complain otherwise.
        sample = self.data[idx][0], torch.Tensor([self.data[idx][1]]).float()

        return sample

def create_data_loaders(data):
    
    dataset = PlanningDataset(data)
    validation_split = 0.2
    batch_size = 128
    # batch_size = 64
    random_seed= 42
    shuffle_dataset = False

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            sampler=train_sampler, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            sampler=valid_sampler, num_workers=2)
                                            
    return [train_loader, valid_loader]

# Neural Network architecture
def get_linear_layer_multiple(value):
    return (value - 5) + 1


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # input channels, output no. of features, kernel size
        self.conv1 = nn.Conv2d(7, 12, 5)
#         self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5)
        # self.fc1 = nn.Linear(16 * 103 * 103, 120)
        # self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc1 = nn.Linear(16 * 33 * 33, 120) # it's 33x33 because the feature maps shrink due to no padding
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv1(x)))
        
#         x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
     #   x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_net(data, epochs, weights_path, net=None):

    train_loader, valid_loader = create_data_loaders(data)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: ", device)
    # condition so that we can retrain an existing network
    if net is None:
        net = Net().to(device)

    # Loss + Optimizer
    criterion = nn.MSELoss()
    # SGD produced too low loss values and forums recommended Adam
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Run network
    start = time.time()
    train_loss_values = list()
    valid_loss_values = list()

    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
        
        # training
        training_loss = 0.0
        net.train()
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # torch.stack() converts list of tensors into a tensor of tensors
            # only then can we apply the to() function
            # inputs, labels = torch.stack(tuple(data[0])).to(device), torch.stack(tuple(data[1])).to(device)
            inputs, labels = torch.stack(tuple(data[0])).to(device), torch.stack(tuple(data[1])).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # REMEMBER TO ADD float()
            outputs = net(inputs.float())
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # print statistics
            training_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                avg_training_loss = training_loss / 100
                print('[%d, %5d] train loss: %.6f' % (epoch + 1, i + 1, avg_training_loss))
                train_loss_values.append(avg_training_loss)
                training_loss = 0.0    

                # validation
                valid_loss = 0.0
                net.eval()   
                for _, data in enumerate(valid_loader, 0):
                    # inputs, labels = data
                    inputs, labels = torch.stack(tuple(data[0])).to(device), torch.stack(tuple(data[1])).to(device)
                    target = net(inputs.float())
                    loss = criterion(target, labels)
                    valid_loss += loss.item()

                avg_valid_loss = valid_loss / len(valid_loader)
                print('[%d, %5d] valid loss: %.6f' % (epoch + 1, i + 1, avg_valid_loss))
                print()
                valid_loss_values.append(avg_valid_loss)

    end = time.time()
    time_taken = (end - start)/60
    print("Time taken: {:.3f}".format(time_taken))

    torch.save(net.state_dict(), weights_path)
    print('Finished Training')

    # plot train and valid loss 
    plt.plot(train_loss_values, label="train loss")
    plt.plot(valid_loss_values, label="valid loss")
    plt.legend(loc='best')
    plt.title("Train Loss vs Valid Loss, time taken: {:.4f}".format(time_taken))
    plt.show()


