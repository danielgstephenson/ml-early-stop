import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_printoptions(sci_mode=False, edgeitems=5)
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)

df = pd.read_csv("cleanData.csv")
df = df[df['choice1']==1]
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

choice1 = df['choice1'].to_numpy()
choice2 = df['choice2'].to_numpy()
happen1 = df['happen1'].to_numpy()

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

# test_losses = cross_validate(data, ntrials=1000, nsteps=600)
# test_losses_df = pd.DataFrame({'test_losses': test_losses})
# test_losses_df.to_csv("test_losses.csv", index=False)

def predict(data, nsteps):
    nobs = data[0].size()[0]
    predictions = np.zeros((nobs, 2))
    for ob in range(nobs):
        print(f"Observation {ob+1}/{nobs}")
        train_data, test_data = leave_one_out(data, ob)
        model = Multilayer_Perceptron().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(nsteps):
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

# os.system('clear')

def estimate_effect(data: tuple[torch.Tensor,torch.Tensor]):
    inputs, targets = data
    predictions = predict(data, nsteps=345) # 345
    treatment = inputs[:,0].cpu().numpy()
    T0 = (treatment == 0) & (choice1 == 1)
    T1 = (treatment == 1) & (choice1 == 1)
    C11 = T0 | T1
    A0 = np.mean(choice2[T0]-predictions[T0,0]) # debiasing term
    A1 = np.mean(choice2[T1]-predictions[T1,1]) # debiasing term
    B0 = np.mean(predictions[C11,0]) 
    B1 = np.mean(predictions[C11,1])
    mu_hat = [A0 + B0, A1 + B1]
    return mu_hat[1] - mu_hat[0]

def shuffle_happen(data: torch.Tensor):
    inputs, targets = data
    shuffled_inputs = inputs.clone()
    shuffled_happen1 = np.random.permutation(happen1)
    shuffled_inputs[:,0] = torch.tensor(shuffled_happen1,dtype=torch.long)
    data = (shuffled_inputs, targets)
    return data

shuffled_data = shuffle_happen(data)
# C11 = choice1 == 1
# C10 = choice1 == 0
# data[0][:,0] - shuffled_data[0][:,0]
# data[0][C11,0] - shuffled_data[0][C11,0]
# data[0][C10,0] - shuffled_data[0][C10,0]

# x = data[0].cpu().numpy()
# y = data[1].cpu().numpy()
# reg = linear_model.LinearRegression()
# reg.fit(x,y)
# reg.coef_

# shuffled_data = shuffle_happen(data)
# sx = shuffled_data[0].cpu().numpy()
# sy = data[1].cpu().numpy()
# sreg = linear_model.LinearRegression()
# sreg.fit(sx,sy)
# sreg.coef_

# Conduct a permutation test:
# Estimate the effect for a shuffled treatment 10,000 times
# For a p-value: percentile of estimated effect with real treatment
# NEXT: Create code to estimate the distribution of shuffled estimates.

effect_estimate = estimate_effect(data)
print(f"Predicted Effect: {effect_estimate}")
permutation_estimates = np.array([effect_estimate])
for step in range(10):
    shuffled_data = shuffle_happen(data)
    estimate = estimate_effect(shuffled_data)
    permutation_estimates = np.append(permutation_estimates, estimate)
    print(f"Shuffled Effect {step+1}: {estimate}")
    np.savetxt('permutation_estimates.csv', 
        permutation_estimates, 
        delimiter=',',
        fmt='%.6f')
    
# Next steps:
# A) Obtain p-value from permutation test
# B) Distribution of effect estimates. Take the average 
# C) Heterogeneity 
# C1) How much heterogeneity exists?
# C2) Which variables are correlated with the Treatment effect? 
#     (e.g. cognition)