using DelimitedFiles
using Flux
using Flux: onehotbatch
using Plots
using ischemia_classification_analysis
using ParametricMachinesDemos
using LineSearches


# Loading and processing data
train = readdlm("data/ECG200_TRAIN.txt")
test = readdlm("data/ECG200_TEST.txt")

y_train = train[:, 1]
y_test = test[:,1]
y_train = onehotbatch(y_train, (-1,1))
y_test = onehotbatch(y_test, (-1,1))

x_train = permutedims(train[:, 2:end], (2,1))
x_test = permutedims(test[:, 2:end], (2,1))

x_train = Flux.unsqueeze(x_train, 2)
x_test = Flux.unsqueeze(x_test, 2)


# Define machine's hyperparameters
machine_type = RecurMachine
dimensions = [16,16,16,16,16,16]
timeblock = 16 
pad = 1 

# Define optimizer's hyperparameters
opt = "Adam"
learning_rate = 0.01
line_search = BackTracking()

# Define training's hyperparameters
n_epochs = 300


# Training

best_params, best_model, loss_on_train, acc_train, acc_test = train_forecast(
    x_train, 
    y_train, 
    x_test, 
    y_test,
    machine_type,
    dimensions, 
    timeblock,
    pad,
    opt, 
    learning_rate, 
    line_search,
    n_epochs, 
    cpu)


# Visualization
plot(1:n_epochs, loss_on_train, lab="Training loss")
yaxis!("Loss");
xaxis!("Training epochs");
savefig("visualization/losses/recurrent/ischemie_rec_loss.png");

plot(1:n_epochs, acc_train, lab="Accuracy on train")
plot!(1:n_epochs, acc_test, lab="Accuracy on test")
yaxis!("Accuracies");
xaxis!("Training epoch");
savefig("visualization/accuracies/recurrent/ischemie_rec_accuracy.png");