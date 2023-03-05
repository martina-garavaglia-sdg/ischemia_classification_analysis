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
embedder = Conv((1,), 1 => 1)

# Loss
smoothness(W) = zero(eltype(W))
smoothness(W, d::Int, ds::Int...) = sum(abs2, diff(W; dims=d)) + smoothness(W, ds...)

time_smoothness(m::RecurMachine) = smoothness(m.W, 1)


loss = function (model, input, output)
    l = crossentropy(model[1](input), output)
    c_t =  1f0 * time_smoothness(model[1][2])
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

best_params, best_model, loss_on_train, acc_train, acc_test = train_forecast(
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

@info "Max accuracy on test:" maximum(acc_test)

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

############################# EXPLAINABILITY ###########################

trained_machine = best_model[2]
trained_embedder = best_model[1]
sensitivity = compute_sensitivity(trained_embedder, trained_machine, x_train)

# test[:,1] .== 1.0
sensitivity_normal = transpose(sensitivity[:,:, 97])
sensitivity_abnormal = transpose(sensitivity[:, :, 6])
serie_normal = x_train[:,:,97]
serie_abnormal = x_train[:,:,6]
plot1 = heatmap(sensitivity_normal, color=:thermal, legend = :none)#, ylab="spaces", xlab="series")
plot2 = heatmap(sensitivity_abnormal, color=:thermal, legend = :none)#, ylab="spaces", lab="series")
plot3 = plot(serie_normal, legend = false)
plot4 = plot(serie_abnormal, legend = false)

plot(plot1, plot2, plot3, plot4)
savefig("visualization/explainability/recc_0.5.png")