# It's empty. Surprise!
# Please complete this by yourself.
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as tfs
import os
import pandas as pd
import copy
import matplotlib.pyplot as plt
from Multi_Network import *
plt.switch_backend('agg')

gpu_id = '1'
epoch = 50
root_dir = '../Dataset/'
train_dir = 'train/'
val_dir = 'val/'
train_anno = 'Multi_train_annotation.csv'
val_anno = 'Multi_val_annotation.csv'
classes = ['Mammls', 'Birds']
species = ['rabbits', 'rats', 'chickens']

class ReadDaraset(data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        if not os.path.isfile(self.data_dir):
            print(self.data_dir + ' does not exist!')
        self.file_info = pd.read_csv(self.data_dir, index_col=0)

    def __getitem__(self, idx):
        image_path = self.file_info['path'][idx]
        if not os.path.isfile(image_path):
            print(image_path + ' does not exist!')
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_class = int(self.file_info['classes'][idx])
        label_species = int(self.file_info['species'][idx])
        return {'image': image, 'classes': label_class, 'species': label_species}

    def __len__(self):
        return len(self.file_info)

train_transforms =  tfs.Compose([
    tfs.Resize((224,224)),
    tfs.RandomHorizontalFlip(),
    tfs.ToTensor(),
    tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = tfs.Compose([
    tfs.Resize((224,224)),
    tfs.ToTensor(),
    tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = ReadDaraset(train_anno, train_transforms)
test_dataset = ReadDaraset(val_anno, test_transforms)

train_dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = data.DataLoader(test_dataset, batch_size=128, shuffle=True)
dataLoaders = {'train': train_dataloader, 'val':  test_dataloader}

device = torch.device('cuda:'+gpu_id if torch.cuda.is_available() else 'cpu')
#print(device)

def train():
    model = MultiClassificationNetWork().to(device)
    #model = Net().to(device)
    optim_params = [{'params': model.class_classifier.parameters(), 'params': model.species_classifier.parameters()}]
    #optim_params = model.parameters()
    optimizer = torch.optim.Adam(optim_params, lr=0.002,betas=(0.5, 0.999))
    criterion = nn.CrossEntropyLoss()

    Loss_list = {'train': [], 'val': []}
    Accuracy_list_species = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for e in range(epoch):
        print('Epoch {}/{}'.format(e, epoch - 1))
        print('-*' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects_species = 0

            for idx, data in enumerate(dataLoaders[phase]):
                # print(phase+' processing: {}th batch.'.format(idx))
                inputs = data['image'].to(device)
                labels_classes = data['classes'].to(device)
                labels_species = data['species'].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(inputs)
                    x_species = output['species'].view(-1, 3)
                    x_classes= output['classes'].view(-1, 2)

                    _, preds_species = torch.max(x_species, 1)
                    _, preds_classes = torch.max(x_classes, 1)

                    loss = criterion(x_species, labels_species) + criterion(x_classes, labels_classes)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                corrects_species += torch.sum(preds_species == labels_species)

            epoch_loss = running_loss / len(dataLoaders[phase].dataset)
            Loss_list[phase].append(epoch_loss)

            epoch_acc_species = corrects_species.double() / len(dataLoaders[phase].dataset)
            epoch_acc = epoch_acc_species

            Accuracy_list_species[phase].append(100 * epoch_acc_species)
            print('{} Loss: {:.4f}  Acc_species: {:.2%}'.format(phase, epoch_loss, epoch_acc_species))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc_species
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best val species Acc: {:.2%}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_model.pt')
    print('Best val species Acc: {:.2%}'.format(best_acc))
    return model, Loss_list, Accuracy_list_species

    # network = Net().to(device)



model, Loss_list, Accuracy_list_classes = train()

x = range(0, epoch)
y1 = Loss_list["train"]
y2 = Loss_list["val"]
#print(len(y1), y1)
plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="train")
plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="val")
plt.legend()
plt.title('train and val loss vs. epoches')
plt.ylabel('loss')
plt.savefig("train and val loss vs epoches.jpg")
plt.close('all')

y5 = Accuracy_list_classes["train"]
y6 = Accuracy_list_classes["val"]
plt.plot(x, y5, color="r", linestyle="-", marker=".", linewidth=1, label="train")
plt.plot(x, y6, color="b", linestyle="-", marker=".", linewidth=1, label="val")
plt.legend()
plt.title('train and val Classes_acc vs. epoches')
plt.ylabel('Classes_accuracy')
plt.savefig("train and val Classes_acc vs epoches.jpg")
plt.close('all')



