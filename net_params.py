import torch
import numpy as np
from torch import nn
from tqdm import trange


class AutoTomo(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(AutoTomo, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        return h


class MNETME(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MNETME, self).__init__()

        self.mnetme = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size), 
            nn.Sigmoid(), 
        )

    def forward(self, x):
        x = self.mnetme(x)
        return x


class RBM:

	def __init__(self, n_visible, n_hidden, lr=0.001, epochs=5, mode='bernoulli', batch_size=32, k=3, optimizer='adam',
				gpu=False, savefile=None, early_stopping_patience=5):
		self.mode = mode  # bernoulli or gaussian RBM
		self.n_hidden = n_hidden  # Number of hidden nodes
		self.n_visible = n_visible  # Number of visible nodes
		self.lr = lr  # Learning rate for the CD algorithm
		self.epochs = epochs  # Number of iterations to run the algorithm for
		self.batch_size = batch_size
		self.k = k
		self.optimizer = optimizer
		self.beta_1 = 0.9
		self.beta_2 = 0.999
		self.epsilon = 1e-07
		self.m = [0, 0, 0]
		self.v = [0, 0, 0]
		self.m_batches = {0: [], 1: [], 2: []}
		self.v_batches = {0: [], 1: [], 2: []}
		self.savefile = savefile
		self.early_stopping_patience = early_stopping_patience
		self.stagnation = 0
		self.previous_loss_before_stagnation = 0
		self.progress = []

		if torch.cuda.is_available() and gpu:
			dev = "cuda:0" 
		else:  
			dev = "cpu"  
		self.device = torch.device(dev)

		# Initialize weights and biases
		std = 4 * np.sqrt(6. / (self.n_visible + self.n_hidden))
		self.W = torch.normal(mean=0, std=std, size=(self.n_hidden, self.n_visible))
		self.vb = torch.zeros(size=(1, self.n_visible), dtype=torch.float32)
		self.hb = torch.zeros(size=(1, self.n_hidden), dtype=torch.float32)

		self.W = self.W.to(self.device)
		self.vb = self.vb.to(self.device)
		self.hb = self.hb.to(self.device)
		
	def sample_h(self, x):
		wx = torch.mm(x, self.W.t())
		activation = wx + self.hb
		p_h_given_v = torch.sigmoid(activation)
		if self.mode == 'bernoulli':
			return p_h_given_v, torch.bernoulli(p_h_given_v)
		else:
			return p_h_given_v, torch.add(p_h_given_v, torch.normal(mean=0, std=1, size=p_h_given_v.shape))

	def sample_v(self, y):
		wy = torch.mm(y, self.W)
		activation = wy + self.vb
		p_v_given_h = torch.sigmoid(activation)
		if self.mode == 'bernoulli':
			return p_v_given_h, torch.bernoulli(p_v_given_h)
		else:
			return p_v_given_h, torch.add(p_v_given_h, torch.normal(mean=0, std=1, size=p_v_given_h.shape))
	
	def adam(self, g, epoch, index):
		self.m[index] = self.beta_1 * self.m[index] + (1 - self.beta_1) * g
		self.v[index] = self.beta_2 * self.v[index] + (1 - self.beta_2) * torch.pow(g, 2)

		m_hat = self.m[index] / (1 - np.power(self.beta_1, epoch)) + (1 - self.beta_1) * g / (1 - np.power(self.beta_1, epoch))
		v_hat = self.v[index] / (1 - np.power(self.beta_2, epoch))
		return m_hat / (torch.sqrt(v_hat) + self.epsilon)

	def update(self, v0, vk, ph0, phk, epoch):
		dW = (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
		dvb = torch.sum((v0 - vk), 0)
		dhb = torch.sum((ph0 - phk), 0)

		if self.optimizer == 'adam':
			dW = self.adam(dW, epoch, 0)
			dvb = self.adam(dvb, epoch, 1)
			dhb = self.adam(dhb, epoch, 2)

		self.W += self.lr * dW
		self.vb += self.lr * dvb
		self.hb += self.lr * dhb

	def train(self, dataset):
		dataset = dataset.to(self.device)
		learning = trange(self.epochs, desc=str('Starting...'))
		for epoch in learning:
			train_loss = 0
			counter = 0
			for batch_start_index in range(0, dataset.shape[0] - self.batch_size, self.batch_size):
				vk = dataset[batch_start_index:batch_start_index + self.batch_size]
				v0 = dataset[batch_start_index:batch_start_index + self.batch_size]
				ph0, _ = self.sample_h(v0)

				for k in range(self.k):
					_, hk = self.sample_h(vk)
					_, vk = self.sample_v(hk)
				phk, _ = self.sample_h(vk)
				self.update(v0, vk, ph0, phk, epoch + 1)
				train_loss += torch.mean(torch.abs(v0 - vk))
				counter += 1

			self.progress.append(train_loss.item() / counter)
			details = {'epoch': epoch + 1, 'loss': round(train_loss.item() / counter, 4)}
			learning.set_description(str(details))
			learning.refresh()

			if train_loss.item() / counter > self.previous_loss_before_stagnation and epoch > self.early_stopping_patience + 1:
				self.stagnation += 1
				if self.stagnation == self.early_stopping_patience - 1:
					learning.close()
					print("Not Improving the stopping training loop.")
					break
			else:
				self.previous_loss_before_stagnation = train_loss.item() / counter
				self.stagnation = 0
		learning.close()
		if self.savefile is not None:
			model = {'W': self.W, 'vb': self.vb, 'hb': self.hb}
			torch.save(model, self.savefile)

	def load_rbm(self, savefile):
		loaded = torch.load(savefile)
		self.W = loaded['W']
		self.vb = loaded['vb']
		self.hb = loaded['hb']

		self.W = self.W.to(self.device)
		self.vb = self.vb.to(self.device)
		self.hb = self.hb.to(self.device)


class DBN(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size,
                 train_data, mode='bernoulli', k=5):
        super(DBN, self).__init__()

        if torch.cuda.is_available():
            dev = "cuda:0" 
        else:  
            dev = "cpu"  
        self.device = torch.device(dev)

        self.input_size = input_size
        self.k = k
        self.mode = mode
        self.layers = [hidden_size1, hidden_size2, hidden_size3]
        self.layer_parameters = [{'W': None, 'hb': None, 'vb': None} for _ in range(len(self.layers))]

        self.train_layer(train_data)

        self.encoder = self.initialize_model()
        for param in self.encoder[:4].parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size3, output_size),
            nn.Sigmoid(),
        )

    def sample_v(self, y, W, vb):
        wy = torch.mm(y, W)
        activation = wy + vb
        p_v_given_h = torch.sigmoid(activation)

        if self.mode == 'bernoulli':
            return p_v_given_h, torch.bernoulli(p_v_given_h)
        else:
            return p_v_given_h, torch.add(p_v_given_h, torch.normal(mean=0, std=1, size=p_v_given_h.shape))

    def sample_h(self, x, W, hb):
        wx = torch.mm(x, W.t())
        activation = wx + hb
        p_h_given_v = torch.sigmoid(activation)

        if self.mode == 'bernoulli':
            return p_h_given_v, torch.bernoulli(p_h_given_v)
        else:
            return p_h_given_v, torch.add(p_h_given_v, torch.normal(mean=0, std=1, size=p_h_given_v.shape))

    def generate_input_for_layer(self, index, x):
        if index > 0:
            x_gen = []

            for _ in range(self.k):
                x_dash = x
                for i in range(index):
                    _, x_dash = self.sample_h(x_dash, self.layer_parameters[i]['W'], self.layer_parameters[i]['hb'])
                x_gen.append(x_dash)

            x_dash = torch.stack(x_gen)
            x_dash = torch.mean(x_dash, dim=0)
        else:
            x_dash = x
        
        return x_dash

    def train_layer(self, x):
        for index, layer in enumerate(self.layers):
            if index == 0:
                vn = self.input_size
            else:
                vn = self.layers[index - 1]
            
            hn = self.layers[index]

            rbm = RBM(vn, hn, epochs=500, mode='bernoulli', lr=1e-3, k=10, batch_size=128, gpu=True, optimizer='adam',
                      early_stopping_patience=20)
            x_dash = self.generate_input_for_layer(index, x)
            rbm.train(x_dash)
            self.layer_parameters[index]['W'] = rbm.W.to(self.device)
            self.layer_parameters[index]['hb'] = rbm.hb.to(self.device)
            self.layer_parameters[index]['vb'] = rbm.vb.to(self.device)
            print("Finished Training Layer:", index, "to", index + 1)

    def initialize_model(self):
        # print("The Last layer will not be activated. The rest are activated using the Sigmoid Function")
        modules = []
        layer_parameters = self.layer_parameters

        for index, layer in enumerate(layer_parameters):
            modules.append(torch.nn.Linear(layer['W'].shape[1], layer['W'].shape[0]))
            if index < len(layer_parameters) - 1:
                modules.append(torch.nn.Sigmoid())

        model = torch.nn.Sequential(*modules)

        for layer_no, layer in enumerate(model):
            if layer_no // 2 == len(layer_parameters) - 1:
                break
            if layer_no % 2 == 0:
                model[layer_no].weight = torch.nn.Parameter(layer_parameters[layer_no // 2]['W'])
                model[layer_no].bias = torch.nn.Parameter(layer_parameters[layer_no // 2]['hb'])
        
        return model

    def forward(self, x):
        x_gen = self.encoder(x)
        output = self.classifier(x_gen)
        return output