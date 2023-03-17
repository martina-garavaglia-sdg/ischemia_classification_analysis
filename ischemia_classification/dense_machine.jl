using DelimitedFiles
using Flux
using Flux: onehotbatch
using ischemia_classification_analysis
using ParametricMachinesDemos
using LineSearches
using Plots

# Loading and processing data
train = readdlm("data/ECG200_TRAIN.txt");
test = readdlm("data/ECG200_TEST.txt");

y_train = train[:,1];
y_test = test[:,1];
y_train = onehotbatch(y_train, (-1,1));
y_test = onehotbatch(y_test, (-1,1));

# dtrain = fit(ZScoreTransform, x_train; dims = 1)
# StatsBase.transform!(dtrain, x_train)
# dtest = fit(ZScoreTransform, x_test; dims = 1)
# StatsBase.transform!(dtest, x_test)

x_train = permutedims(train[:, 2:end], (2,1));
x_test = permutedims(test[:, 2:end], (2,1));


# Define machine's hyperparameters
machine_type = DenseMachine
dimensions = [64,32,16,8] # solo 64,8
timeblock = 0 # only for recurrent
pad = 0 # only for recurrent
embedder = Dense(96,64)

# Loss
smoothness(W) = zero(eltype(W))
smoothness(W, d::Int, ds::Int...) = sum(abs2, diff(W; dims=d)) + smoothness(W, ds...)

time_smoothness(m::DenseMachine) = smoothness(m.W, 1)


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

best_params, best_model, loss_on_train, acc_train, acc_test = train_classification(
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
    device);

@info "Max accuracy on test:" maximum(acc_test)


# # Visualization
plot(1:n_epochs, loss_on_train, xtickfontsize=10, ytickfontsize=10, xguidefontsize=10, yguidefontsize=10, legendfontsize=12, lw=2, lab="Training loss", ylim=(0,1))
yaxis!("Loss on train");
xaxis!("Training epochs");
savefig("visualization/losses/ischemie_dense_loss_reg.png");

plot(1:n_epochs, acc_train, lab="Accuracy on train", xtickfontsize=10, ytickfontsize=10, xguidefontsize=10, yguidefontsize=10, legendfontsize=12, lw=2, ylim=(0,1))
plot!(1:n_epochs, acc_test, lab="Accuracy on test", xtickfontsize=10, ytickfontsize=10, xguidefontsize=10, yguidefontsize=10, legendfontsize=12, lw=2, ylim=(0,1))
yaxis!("Accuracies");
xaxis!("Training epoch");
savefig("visualization/accuracies/ischemie_dense_accuracy_reg.png");


###################### EXPLAINABILITY #################

# trained_machine = best_model[2]
# trained_embedder = best_model[1]
# sensitivity = compute_sensitivity(trained_embedder, trained_machine, x_test)

# sensitivity_normal = sensitivity[:,test[:,1] .== -1.0]
# sensitivity_abnormal = sensitivity[:,test[:,1] .== 1.0]
# plot1 = heatmap(sensitivity_normal, color=:thermal, xlab = "Samples (normal)", ylab = "Depth", legend =:none)
# plot2 = heatmap(sensitivity_abnormal, color=:thermal, xlab = "Samples (abnormal)", ylab = "Depth", legend =:none)

# plot(plot1, plot2)
# savefig("visualization/explainability/dense1.png")
