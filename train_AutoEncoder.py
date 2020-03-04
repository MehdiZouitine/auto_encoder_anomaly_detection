import os
import random
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
# specify loss function
criterion = nn.L1Loss()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = AutoEncoder()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
data_train = DatasetAutoEncoder(
      csv_folder='/content/drive/My Drive/data', split='train',norm=False)
train_loader = data.DataLoader(
    data_train, batch_size=32, shuffle=True, num_workers=4)

seed_everything(42)

n_epochs = 30

for epoch in tqdm(range(1, n_epochs+1)):
    # monitor training loss
    train_loss = 0.0
    ###################
    # train the model #
    ###################
    for data in train_loader:
        # _ stands in for labels, here
        # no need to flatten images
        data = data.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(data)
        # calculate the loss
        loss = criterion(outputs, data)
        #print(loss.item())
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item() * data.size(0)

    # print avg training statistics
    train_loss = train_loss/len(data_train)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch,
        train_loss
        ))

def auto_encoder_predict(ds,criterion,size):
  loss_pred = []
  for i in range (size):
    x = ds[i].view(1,1,61440).to(device)
    y = model(x)
    loss_pred.append(criterion(x,y).item())
  return loss_pred
