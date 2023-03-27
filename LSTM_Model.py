from torch.utils.data import DataLoader
from torch import nn, optim
import random as rand
import numpy as np
import torch
import math
import time



class Model(nn.Module):

    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size = 40
        self.embedding_dim = 40
        self.num_layers = 1

        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))


def train(dataset, model):
    model.train()

    dataloader = DataLoader(dataset, batch_size=80)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(1):
        start_epoch = time.time()
        state_h, state_c = model.init_state(40)

        for batch, (x, y) in enumerate(dataloader):
            start_batch = time.time()
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            end_batch = time.time()
            if not batch % 10:
                print({'epoch': epoch, 'batch': batch})
                print('Loss:','{0:.3f}'.format(loss.item()),'Batch_Time:', '{0:.3f}'.format((end_batch - start_batch)))
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'.format(epoch, batch * len(x), len(dataloader.dataset),
                       100. * batch / len(dataloader)))

        end_epoch = time.time()
        print('Epoch_Time:', '{0:.3f}'.format((end_epoch - start_epoch)/60))


def generate_sentence(dataset, model, text, sentence=40):
    model.eval()

    words = text.split()
    state_h, state_c = model.init_state(len(words))

    for i in range(0, sentence):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)

        words.append(dataset.index_to_word[word_index])

    return ' '.join(words)
def last_word_logits(dataset, model, idx_batch,cuda=False):
    model.eval()

    state_h, state_c = model.init_state(len(idx_batch[0]))

 
    if cuda:
      y_pred, (state_h, state_c) = model(idx_batch, (state_h.cuda(), state_c.cuda()))  
    else:
      y_pred, (state_h, state_c) = model(idx_batch, (state_h, state_c) )

    last_words=[]
    #print(logit)
    for i in y_pred:
      last_words.append(list(i[-1]))

    return torch.tensor(last_words)

   


def word_sampling(dataset):
    words = []

    while len(words) != 1000:
        word = dataset.uniq_words[rand.randint(0, len(dataset.uniq_words))]

        if len(word) > 3:
            words.append(word)

    return words

def generate_1k_sentences(dataset, model):

    words = word_sampling(dataset)
    phrases = [generate_sentence(dataset, model, el)+'\n' for el in words]

    with open('LSTM_1k_sentences.txt', 'w', encoding='utf8') as f:
        f.writelines(phrases)


# =====================================================================================================================
#
# #batch_size = 40
#
# def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
#     """ Generate a word from from out[gen_idx]"""
#
#     logits = out[:, gen_idx]
#     if temperature is not None:
#         logits = logits / temperature
#     if top_k > 0:
#         kth_vals, kth_idx = logits.topk(top_k, dim=-1)
#         dist = torch.distributions.categorical.Categorical(logits=kth_vals)
#         idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
#     elif sample:
#         dist = torch.distributions.categorical.Categorical(logits=logits)
#         idx = dist.sample().squeeze(-1)
#     else:
#         idx = torch.argmax(logits, dim=-1)
#     return idx.tolist() if return_list else idx
#
# def tokenize_batch(batch):
#     return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]
#
# def get_init_text(seed_text, max_len, batch_size=1, rand_init=False):
#     """ Get initial sentence by padding seed_text with either masks or random words to max_len """
#     batch = [[CLS] + seed_text.split() + [] * max_len + [SEP] for _ in range(batch_size)]
#     return tokenize_batch(batch)
#
# def untokenize_batch(batch):
#     return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]
#
# def parallel_sequential_generation(model, seed_text, batch_size, max_len=40, top_k=0, temperature=None, max_iter=300, burnin=200,
#                                    cuda=False, print_every=10, verbose=True):
#     """ Generate for one random position at a timestep
#     args:
#         - burnin: during burn-in period, sample from full distribution; afterwards take argmax
#     """
#     seed_len = len(seed_text)
#     print(seed_text, '\n', seed_len)
#     batch = get_init_text(seed_text, max_len, batch_size)
#
#     state_h, state_c = model.init_state(len(seed_text))
#
#     for ii in range(max_iter):
#         kk = np.random.randint(0, max_len)
#         # print(kk, type(kk))
#         # for jj in range(batch_size):
#         #     print(batch, kk)
#         #     batch[jj][seed_len + kk] = mask_id
#         # inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
#         out = model(batch, (state_h, state_c))
#         topk = top_k if (ii >= burnin) else 0
#         idxs = generate_step(out, gen_idx=seed_len + kk, top_k=topk, temperature=temperature, sample=(ii < burnin))
#         for jj in range(batch_size):
#             batch[jj][seed_len + kk] = idxs[jj]
#
#         if verbose and np.mod(ii + 1, print_every) == 0:
#             for_print = tokenizer.convert_ids_to_tokens(batch[0])
#             for_print = for_print[:seed_len + kk + 1] + ['(*)'] + for_print[seed_len + kk + 1:]
#             print("iter", ii + 1, " ".join(for_print))
#
#     return untokenize_batch(batch)
#
#
# def generate(model, n_samples, seed_text="[CLS]", batch_size=10, max_len=25,
#              sample=True, top_k=100, temperature=1.0, burnin=200, max_iter=500,
#              cuda=False, print_every=1):
#     # main generation function to call
#     sentences = []
#     n_batches = math.ceil(n_samples / batch_size)
#     start_time = time.time()
#     for batch_n in range(n_batches):
#         batch = parallel_sequential_generation(model, seed_text, batch_size, max_len=max_len, top_k=top_k,
#                                                temperature=temperature, burnin=burnin, max_iter=max_iter,
#                                                cuda=cuda, verbose=False)
#
#         if (batch_n + 1) % print_every == 0:
#             print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
#             start_time = time.time()
#
#         sentences += batch
#     return sentences