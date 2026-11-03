# Reference model definitions for HJNC Final Modeling v05.
# Preprocessing must follow model_config.json and MODEL_TRAINING_REPORT_KO.md.
import torch
import torch.nn as nn

class RouteMLP(nn.Module):
    def __init__(self, vocab=1008, nclass=1007, emb=32, hidden=128, ctx=8, padding_idx=1007):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=padding_idx)
        self.net = nn.Sequential(nn.Linear(ctx*emb, hidden), nn.ReLU(), nn.Dropout(.1), nn.Linear(hidden, nclass))
    def forward(self, x):
        return self.net(self.emb(x).reshape(x.size(0), -1))

class RouteRNN(nn.Module):
    def __init__(self, kind='GRU', vocab=1008, nclass=1007, emb=24, hidden=48, padding_idx=1007):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=padding_idx)
        R = nn.GRU if kind == 'GRU' else nn.LSTM
        self.rnn = R(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, nclass)
        self.kind = kind
    def forward(self, x):
        _, h = self.rnn(self.emb(x))
        if self.kind == 'LSTM':
            h = h[0]
        return self.out(h[-1])

class ForecastRNN(nn.Module):
    def __init__(self, kind='GRU', fin=4, nlinks=1007, emb=16, hidden=64):
        super().__init__()
        R = nn.GRU if kind == 'GRU' else nn.LSTM
        self.rnn = R(fin, hidden, batch_first=True)
        self.linkemb = nn.Embedding(nlinks, emb)
        self.kind = kind
        self.head = nn.Sequential(nn.Linear(hidden+emb+2, 96), nn.ReLU(), nn.Dropout(.1), nn.Linear(96, 3))
    def forward(self, x, link, timef):
        _, h = self.rnn(x)
        if self.kind == 'LSTM':
            h = h[0]
        return self.head(torch.cat([h[-1], self.linkemb(link), timef], dim=1))

class GraphGRU(nn.Module):
    # fin=5: own_share, A_norm*share, run_share, log_duration, log_delay
    def __init__(self, nlinks=1007, emb=16, hidden=64):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(5, 24), nn.ReLU())
        self.gru = nn.GRU(24, hidden, batch_first=True)
        self.linkemb = nn.Embedding(nlinks, emb)
        self.head = nn.Sequential(nn.Linear(hidden+emb+2, 96), nn.ReLU(), nn.Dropout(.1), nn.Linear(96, 3))
    def forward(self, x, link, timef):
        _, h = self.gru(self.pre(x))
        return self.head(torch.cat([h[-1], self.linkemb(link), timef], dim=1))
