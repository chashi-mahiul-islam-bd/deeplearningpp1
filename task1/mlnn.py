'''' NETWORK ARCHITECTURE '''
import torch 
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
import pa1_dataloader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# architecture

train_loader = pa1_dataloader.load(batch_size = 1, shuffle=True)

class Classifier(nn.Module):
    def __init__(self, hidden_1, hidden_2, hidden_3, dropout_prob):
        super(Classifier, self).__init__()
        
        self.fc1 = nn.Linear(16*16, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.fc4 = nn.Linear(hidden_3, 10)
        
        self.dropout = nn.Dropout(p=dropout_prob)
        
    def forward(self, x):
        # flattening the input tensor
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

'''' MODEL, LOSS FUNCTION, AND OPTIMIZER '''

model = Classifier(hidden_1=256, hidden_2=128, hidden_3=64, dropout_prob=0.2)
print(model)

# =============================================================================
# if train_on_gpu:
#     model.cuda()
# =============================================================================

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
#optimizer = optim.SGD(model.parameters(), lr=0.003)


'''' TRAINING AND VALIDATION '''
# training
epochs = 10

# keeps minimum validation loss
valid_loss_min = np.Inf

train_losses, test_losses = [], []
for epoch in range(epochs):
    
    train_loss = 0
    
    # training the model
    
    model.train()
    for images, labels in train_loader:
# =============================================================================
#         if train_on_gpu:
#             images, labels = images.cuda(), labels.cuda()
# =============================================================================
            
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item()
        print("Epoch: {}/{}.. ".format(epoch+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss/len(train_loader)))
    '''
    else:
        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
           for images, labels in test_loader:
               # =============================================================================
               #         if train_on_gpu:
               #             images, labels = images.cuda(), labels.cuda()
               # =============================================================================
               output = model(images)
               valid_loss += criterion(output, labels)
               
               ps = torch.exp(output)
               top_p, top_class = ps.topk(1, dim=1)
               equals = top_class == labels.view(*top_class.shape)
               accuracy += torch.mean(equals.type(torch.FloatTensor))
       
        train_losses.append(train_loss/len(train_loader))
        test_losses.append(valid_loss/len(test_loader))
        
        print("Epoch: {}/{}.. ".format(epoch+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss/len(train_loader)),
              "Validation Loss: {:.3f}.. ".format(valid_loss/len(test_loader)),
              "Validation Accuracy: {:.3f}".format(accuracy/len(test_loader)))
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving Model ...'.format(
                    valid_loss_min,
                    valid_loss))
            torch.save(model.state_dict(), 'mlp_model_Adam_fasion_mnist.pt')
            valid_loss_min = valid_loss
        model.train()
    '''