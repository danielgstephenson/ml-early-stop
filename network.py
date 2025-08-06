import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_printoptions(sci_mode=False, edgeitems=5)
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)

df = pd.read_csv("cleanData.csv")
force_category_columns = [
	'cognition1',
	'cognition2'
]
for column_name, dtype in df.dtypes.items():
	if dtype != 'int64' or column_name in force_category_columns:
		df[column_name] = df[column_name].astype('category').cat.codes

drop_columns = [
	'subject',
	'choice2',
	'happen2',
	'possible2',
	'experimentTime',
	'timeTaken',
	'engagement',
	'clicks',
	'completionCode',
	'possible1',
	'quizEndTime',
	'surveyEndTime'
]
input_df = df.drop(drop_columns,axis=1)

treatment_df = input_df['happen1']
treatment_inputs = torch.tensor(treatment_df.to_numpy(),dtype=torch.long).unsqueeze(1)

category_column_names = [
	'choice1',
	'ethnicity',
	'nation',
	'sex',
	'employment',
	'studentStatus',
	'cognition1',
	'cognition2'
]
category_df = input_df[category_column_names]
category_ints = torch.tensor(category_df.to_numpy(),dtype=torch.long)
category_one_hots = [F.one_hot(category_ints[:,j]) for j in range(category_ints.size()[1])]
category_inputs = torch.cat(category_one_hots,1)

continuous_column_names = [
	'surveyTime',
	'quizTime',
	'approvals',
	'age'
]
continuous_df = input_df[continuous_column_names]
continuous_tensor = torch.tensor(continuous_df.to_numpy(),dtype=torch.float)
continuous_columns = [continuous_tensor[:,j] for j in range(continuous_tensor.size()[1])]
standardized_columns = [
	torch.unsqueeze((c - torch.mean(c)) / torch.std(c),1)
	for c in continuous_columns
]
standardized_inputs = torch.cat(standardized_columns,1)

inputs = torch.cat([treatment_inputs, category_inputs, standardized_inputs],1).to(device)
targets = torch.tensor(df['choice2'].to_numpy(),dtype=torch.float).to(device)
data = (inputs, targets)

n1 = 5 # number of neurons in first hidden layer
n2 = 5 # number of neurons in second hidden layer

class Multilayer_Perceptron(torch.nn.Module):
	def __init__(self):
		super(Multilayer_Perceptron,self).__init__()
		self.input_layer = torch.nn.Linear(inputs.size()[1],n1)
		self.elu = torch.nn.ELU()
		self.sigma = torch.nn.Sigmoid()
		self.hidden1 = torch.nn.Linear(n1,n2)
		self.hidden2 = torch.nn.Linear(n2,1)
	def forward(self,x):
		x = self.input_layer(x)        
		x = self.elu(x)
		x = self.hidden1(x)
		x = self.elu(x)
		x = self.hidden2(x)
		x = self.sigma(x)
		output = x.squeeze(1) if x.dim() > 1 else x
		return output
	
loss_function = torch.nn.MSELoss()

def test_step(model, data):
	model.eval()
	with torch.no_grad():
		x,y = data
		outputs = model(x)
		if outputs.dim() == 0:
			outputs = outputs.unsqueeze(1)
		loss = loss_function(outputs,y)
		return loss.item()

def train_step(model, data, optimizer):
	model.train()
	optimizer.zero_grad()
	x, y = data
	predictions = model(x)
	loss = loss_function(predictions, y)
	loss.backward()
	optimizer.step()
	return loss.item()

def leave_one_out (data, observation):
	x, y = data
	test_x = x[observation].unsqueeze(0)
	test_y = y[observation].unsqueeze(0)
	train_x = torch.cat((x[:observation], x[observation+1:]), dim=0)
	train_y = torch.cat((y[:observation], y[observation+1:]), dim=0)
	return (train_x, train_y), (test_x, test_y)

def cross_validate(data, ntrials=1000, nsteps=1000):
	steps = [i for i in range(nsteps)]
	nobs = data[0].size()[0]
	test_losses = np.zeros((ntrials, nsteps))
	plt.ion()
	for trial in range(ntrials):
		print(f"Trial {trial+1}/{ntrials}")
		ob = np.random.randint(0, nobs)
		train_data, test_data = leave_one_out(data, ob)
		model = Multilayer_Perceptron().to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
		for step in range(nsteps):
			train_step(model, train_data, optimizer)
			test_loss = test_step(model, test_data)
			test_losses[trial, step] = test_loss
		mean_losses = np.mean(test_losses[:(trial+1), :], 0)
		plt.clf()
		plt.plot(steps, mean_losses, label='Test Loss')
		plt.pause(0.1)
	mean_losses = np.mean(test_losses, 0)
	stop_step = np.argmin(mean_losses)
	print(f"stop_step: {stop_step}")
	return mean_losses
	
def predict(data, nsteps=100):
	nobs = data[0].size()[0]
	predictions = np.zeros((nobs, 2))
	for ob in range(nobs):
		print(f"Observation {ob+1}/{nobs}")
		train_data, test_data = leave_one_out(data, ob)
		model = Multilayer_Perceptron().to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
		for step in range(nsteps):
			train_step(model, train_data, optimizer)
		input0 = test_data[0].clone()
		input1 = test_data[0].clone()
		input0[0,0] = 0
		input1[0,0] = 1
		model.eval()
		with torch.no_grad():
			output0 = model(input0)
			output1 = model(input1)
		predictions[ob, 0] = output0.item()
		predictions[ob, 1] = output1.item()
	return predictions

# test_losses = cross_validate(data, ntrials=1000, nsteps=600)
# test_losses_df = pd.DataFrame({'test_losses': test_losses})
# test_losses_df.to_csv("test_losses.csv", index=False)

predictions = predict(data, nsteps=300)
print(predictions)
predicted_intercept = predictions[:,0]
predicted_treatment_effect = predictions[:,1] - predictions[:,0]
plt.clf()
plt.ion()
hist = plt.hist(predicted_treatment_effect, bins=20)
predict_ate = np.mean(predicted_treatment_effect)
print(f"Predicted ATE: {predict_ate}")

choice1 = df['choice1'].to_numpy()
choice2 = df['choice2'].to_numpy()
happen1 = df['happen1'].to_numpy()
ml_data = pd.DataFrame({
	'choice1': choice1,
	'choice2': choice2,
	'happen1': happen1,
	'ML0': predicted_intercept,
	'ML1': predicted_treatment_effect
})
ml_data.to_csv("ml_data.csv", index=False)

