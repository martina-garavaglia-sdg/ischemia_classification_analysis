using DelimitedFiles
using Flux
using Flux: onehotbatch, crossentropy, onecold
using Plots
using ischemia_classification_analysis
using ParametricMachinesDemos
using LineSearches
using StatsBase
using Gadfly
using BSON: @load, @save


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
embedder = Conv((1,), 1 => 16)

# Loss
smoothness(W) = zero(eltype(W))
smoothness(W, d::Int, ds::Int...) = sum(abs2, diff(W; dims=d)) + smoothness(W, ds...)

time_smoothness(m::RecurMachine) = smoothness(m.W, 1)


loss = function (model, input, output)
    l = crossentropy(model(input), output)
    c_t =  0.01f0 * time_smoothness(model[2])
    return l + c_t
end


# Define optimizer's hyperparameters
opt = "Adam"
learning_rate = 0.01
line_search = BackTracking()

# Define training's hyperparameters
n_epochs = 300
device = cpu


# Training

best_params_reg, best_model_reg, loss_on_train_reg, acc_train_reg, acc_test_reg = train_classification(
    x_train, 
    y_train, 
    x_test, 
    y_test,
    machine_type,
    dimensions,
    loss;
    embedder,
    timeblock,
    pad,
    opt, 
    learning_rate, 
    line_search,
    n_epochs, 
    device)

@info "Max accuracy on test:" maximum(acc_test_reg)

# @save "mymodel.bson" best_model
# #using Flux: flatten

# Visualization
Plots.plot(1:n_epochs, loss_on_train, xtickfontsize=10, ytickfontsize=10, xguidefontsize=10, yguidefontsize=10,color="blue", legend=:none, lw=2, ylim=(0,1), lab="Train loss")
Plots.plot!(1:n_epochs, loss_on_train_reg, color="blue", lw=2, alpha=0.5, ls=:dots, lab="Regularized train loss")
yaxis!("Loss");
xaxis!("Training epochs");
savefig("visualization/losses/ischemie_RECURRENT_loss.png");

Plots.plot(1:n_epochs, acc_train, xtickfontsize=10, ytickfontsize=10, xguidefontsize=10, yguidefontsize=10, legend=:none, ylim=(0,1), lw=2, color="blue", lab="Train accuracy")
Plots.plot!(1:n_epochs, acc_test, xtickfontsize=10, ytickfontsize=10, xguidefontsize=10, yguidefontsize=10, legendfontsize=12, ylim=(0,1), lw=2, color="green", lab="Test accuracy")
Plots.plot!(1:n_epochs, acc_train_reg, color="blue", lw=2, alpha=0.5,ls=:dot, lab="Regularized train accuracy")
Plots.plot!(1:n_epochs, acc_test_reg, color="green", lw=2, alpha=0.5,ls=:dot, lab="Regularized test accuracy")
yaxis!("Accuracies");
xaxis!("Training epochs");
savefig("visualization/accuracies/ischemie_RECURRENT_accuracy.png");
