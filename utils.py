import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook, tqdm
import torch
import torch.nn.functional as F


def load_pretrained_vectors(word2idx, filename):
    """Load pretrained vectors and create embedding layers.
    
    Args:
        word2idx (Dict): Vocabulary built from the corpus
        filename (str): Path to pretrained vector file

    Returns:
        embeddings (np.array): Embedding matrix with shape (N, d) where N is
            the size of word2idx and d is embedding dimension
    """

    print("Loading pretrained vectors...")
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())

    # Initilize random embeddings
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), d))
    embeddings[word2idx['<pad>']] = np.zeros((d,))
    embeddings[word2idx['<unk>']] = np.zeros((d,))
    
    print(max(word2idx.values()))

    # Load pretrained vectors
    count = 0
    for line in tqdm_notebook(fin):
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word in word2idx:
            count += 1
            embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)

    print(f"There are {count} / {len(word2idx)} pretrained vectors found.")

    return embeddings


def sig(x):
    return F.sigmoid(x)


def train_loop(dataloader, model, loss_fn, optimizer, device):
    """
    Perform one epoch of training through the dataset.
    """
    num_batches = len(dataloader)
    total_loss = 0
    correct_cases = 0
    precision = 0
    recall = 0
    # counter for printing training loss
    # cnt = 0

    model.train()
    for X, y in dataloader:
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
    
        pred = model(X)
        pred = torch.squeeze(pred)
        # squeeze to match the dimensions of pred and y
        loss = loss_fn(pred, y.float())
        pred = sig(pred)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_cases += torch.sum((pred > 0.5) == (y>0)).item()/X.shape[0]
        # if cnt % 100:
        #    print(f"Current avg loss:{total_loss/(cnt+1)}\n")
        # cnt += 1   
    # print(f"Training loss: {total_loss/num_batches:>5f}")
    
    return total_loss/num_batches, correct_cases/num_batches


def test_loop(dataloader, model, loss_fn, device, output_stats=False):
    """
    Perform one iteration of test loop
    Return:
    - loss
    - accuracy
    (optional)
    - precision
    - recall
    """

    num_batches = len(dataloader)
    total_loss = 0
    correct_cases = 0
    true_pos = 0
    total_pos = 0
    actual_pos = 0
    true_neg = 0
    total_neg = 0
    actual_neg = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = torch.squeeze(pred)
            # squeeze to match the dimensions of pred and y
            total_loss += loss_fn(pred, y.float()).item()
            pred = sig(pred)
            correct_cases += torch.sum((pred>0.5) == (y>0)).item()/X.shape[0]
            true_pos += torch.sum((pred>0.5) & (y>0)).item()
            total_pos += torch.sum((pred>0.5)).item()
            actual_pos += torch.sum((y>0)).item()

            true_neg += torch.sum((pred<=0.5) & (y==0)).item()
            total_neg += torch.sum((pred<=0.5)).item()
            actual_neg += torch.sum((y==0)).item()

    if output_stats:
        precision_ham = true_pos/total_pos
        recall_ham = true_pos/actual_pos

        precision_spam = true_neg/total_neg
        recall_spam = true_neg/actual_neg

        return total_loss/num_batches, correct_cases/num_batches, precision_ham, recall_ham, precision_spam, recall_spam
    else:
        return total_loss/num_batches, correct_cases/num_batches


def train_full_test_once(train_dataloader, test_dataloader, model, loss_fn, optimizer, task_name,
                         device="cuda", epochs=100, vis=True, print_every=5, img_dir=""):
    """
    Perform `epochs` loops of training and test the model once. Returns the final results of training 
    and testing.
    Args:
        training_dataloader:
            Training set dataloader.
        testing_data:
            Testing set dataloader.
        model:
            Machine learning model.
        loss_fn:
            Loss function of the model.
        optimizer:
            Optimizer of the model.
        device:
            Device this model is trained on.
        epochs (int):
            The number of rounds of which the dataset is fed into the network.
        print_every (int):
            Frequency to print training loss.
        img_dir (string):
            Plots will be saved under this directory
    Returns:
        train_values (ndarray of shape (n_params, )):
            Training results.
        test_values (ndarray of shape (n_params, )):
            Testing results.
    """    

    loss_list = []
    acc_list = []

    for t in tqdm(range(epochs)):
        train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loss, test_acc = test_loop(test_dataloader, model, loss_fn, device) 

        loss_list.append(train_loss)
        acc_list.append(train_acc)


        if t % print_every == 0:
            print(f"Epoch {t}\n-------------------------------")
            print(f"Training loss: {train_loss:>5f}, avg accuracy: {train_acc:>3f}")

    if vis:
        plot_loss(epochs, loss_list, title=task_name+" loss", filename=img_dir+task_name+"train_loss")
        plot_acc(epochs, acc_list, title=task_name+" accuracy", filename=img_dir+task_name+"train_acc")

    test_loss, test_acc, precision, recall = test_loop(test_dataloader, model, loss_fn, device, output_stats=True)
    print(f"Final testing loss: {test_loss:>5f}, testing accuracy: {test_acc:>3f}")
    print(f"F1 score: {2*precision*recall/(precision+recall):3f} Precision: {precision:3f}, Recall: {recall:3f}")

    return train_loss, test_loss


def train_test_scheme(train_dataloader, test_dataloader, model, loss_fn, optimizer, task_name,
                      device="cuda", epochs=100, vis=True, print_every=5, img_dir=""):
    """
    Perform `epochs` loops of training and test the model once. Returns the final results of training 
    and testing.
    Args:
        training_dataloader:
            Training set dataloader.
        testing_data:
            Testing set dataloader.
        model:
            Machine learning model.
        loss_fn:
            Loss function of the model.
        optimizer:
            Optimizer of the model.
        device:
            Device this model is trained on.
        epochs (int):
            The number of rounds of which the dataset is fed into the network.
        print_every (int):
            Frequency to print training loss.
        img_dir (string):
            Plots will be saved under this directory
    Returns:
        train_values (ndarray of shape (n_params, )):
            Training results.
        test_values (ndarray of shape (n_params, )):
            Testing results.
    """    

    loss_list = []
    acc_list = []

    test_loss_list = []
    test_acc_list = []

    for t in tqdm(range(epochs)):
        train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loss, test_acc = test_loop(test_dataloader, model, loss_fn, device) 

        loss_list.append(train_loss)
        acc_list.append(train_acc)

        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        if t % print_every == 0:
            print(f"Epoch {t}\n-------------------------------")
            print(f"Training loss: {train_loss:>5f}, avg accuracy: {train_acc:>3f}")
            print(f"Testing loss: {test_loss:>5f}, avg accuracy: {test_acc:>3f}")

    if vis:
        plot_train_test_loss(epochs, loss_list, test_loss_list, title=task_name+" loss", filename=img_dir+task_name+"train_loss")
        plot_train_test_acc(epochs, acc_list, test_acc_list, title=task_name+" accuracy", filename=img_dir+task_name+"train_acc")

    test_loss, test_acc, precision_ham, recall_ham, precision_spam, recall_spam = test_loop(test_dataloader, model, loss_fn, device, output_stats=True)
    print(f"Final testing loss: {test_loss:>5f}, testing accuracy: {test_acc:>3f}")
    print(f"Ham Precision: {precision_ham:3f}, Recall: {recall_ham:3f}, F1 score: {2*precision_ham*recall_ham/(precision_ham+recall_ham):3f}.")
    print(f"Ham Precision: {precision_spam:3f}, Recall: {recall_spam:3f}, F1 score: {2*precision_spam*recall_spam/(precision_spam+recall_spam):3f}.")


    return train_loss, test_loss

# TODO: modify the functions after this line

# def cross_validate_scheme(whole_dataset, model_class, loss_fn,
#                           lr=0.01, batch_size=64, reg=0, epochs=10, k=5, 
#                           device="cpu"):
#     """
#     Perform k-fold cross validation. The averaged training and validating results in each epoch 
#     are stored in `train_results` and `valid_results`.

#     Args:
#         whole_dataset (PyTorch DataSet class):
#             The entire dataset ready to split into training and testing sets.
#         model_class:
#             The class of the machine learning model.
#         loss_fn:
#             The loss function of the model.
#         lr (float):
#             Learning rate of the model.
#         batch_size (int):
#             The size of samples fed into the network in each training iteration.
#         epochs (int):
#             The number of rounds of which the dataset is fed into the network.

#     Returns:
#         train_results (ndarray of shape (epochs, n_params)):
#             Training results.
#         valid_results (ndarray of shape (epochs, n_params)):
#             Validation results.
#     """

#     n = len(whole_dataset)
#     idx = np.arange(n)
#     np.random.seed(0)
#     np.random.shuffle(idx)
    
#     data, labels = whole_dataset[idx,:]

#     train_results = np.zeros((epochs, 5))
#     valid_results = np.zeros((epochs, 5))

#     s_data = np.array_split(data, k, axis=0)
#     s_labels = np.array_split(labels, k, axis=0)
#     print(f"{k}-fold cross validation\n-------------------------------")
#     for i in range(k):
#         model = model_class().to(device)
#         # Change the following line to use a different optimizer
#         optimizer = Adam(model.parameters(), lr=lr, weight_decay=reg)

#         data_train = np.concatenate(s_data[:i] + s_data[i+1:], axis=0)
#         labels_train = np.concatenate(s_labels[:i] + s_labels[i+1:], axis=0)
        
#         data_valid = s_data[i]
#         labels_valid = s_labels[i]

#         train_set = TensorDataset(torch.from_numpy(data_train).float(), torch.from_numpy(labels_train).float())
#         valid_set = TensorDataset(torch.from_numpy(data_valid).float(), torch.from_numpy(labels_valid).float())

#         print(f"Fold {i+1}\n-------------------------------")
#         train_values, valid_values = train_test_scheme(train_set, valid_set, model, loss_fn, 
#                                                    optimizer=optimizer, 
#                                                    batch_size=batch_size, 
#                                                    epochs=epochs,
#                                                    device=device)

#         train_results += train_values
#         valid_results += valid_values

#     train_results, valid_results = train_results/k, valid_results/k

#     return train_results, valid_results

def plot_train_test_loss(epochs, train_loss, test_loss, title="", filename=None):

    # Plot epoch loss
    plt.figure(facecolor="white")
    plt.plot(range(epochs), train_loss, label='training')
    plt.plot(range(epochs), test_loss, label='testing')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(title)
    plt.legend()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def plot_loss(epochs, train_loss, title="", filename=None):

    # Plot epoch loss
    plt.figure(facecolor="white")
    plt.plot(range(epochs), train_loss, label='training')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(title)
    plt.legend()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def plot_train_test_acc(epochs, train_acc, test_acc, title="", filename=None):

    # Plot epoch loss
    plt.figure(facecolor="white")
    plt.plot(range(epochs), train_acc, label='training')
    plt.plot(range(epochs), test_acc, label='testing')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title(title)
    plt.legend()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def plot_acc(epochs, train_acc, title="", filename=None):

    # Plot epoch loss
    plt.figure(facecolor="white")
    plt.plot(range(epochs), train_acc, label='training')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title(title)
    plt.legend()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
