using DelimitedFiles
using Flux
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten
using Statistics: mean, std
using Flux.Data: DataLoader
using Plots
using ParametricMachinesDemos
using ischemia_classification_analysis


train = readdlm("ischemie/data/ECG200_TRAIN.txt")
test = readdlm("ischemie/data/ECG200_TEST.txt")

y_train = train[:, 1]
y_test = test[:,1]
y_train = onehotbatch(y_train, (-1,1))
y_test = onehotbatch(y_test, (-1,1))

x_train = permutedims(train[:, 2:end], (2,1))
x_test = permutedims(test[:, 2:end], (2,1))

x_train = Flux.unsqueeze(x_train, 1)
x_test = Flux.unsqueeze(x_test, 1)
x_train = Flux.unsqueeze(x_train, 3)
x_test = Flux.unsqueeze(x_test, 3)


# Loading
train_data = DataLoader((x_train, y_train); batchsize = 32, shuffle = true);
test_data = DataLoader((x_test, y_test); batchsize = 32, shuffle = true);

dimensions = [2,2,2,2];

# Define the parametric machine
machine = ConvMachine(dimensions, sigmoid; pad=(0,0,0,0))

# Model
# model = Flux.Chain(Conv((1,), 1 => 8), machine, flatten, Dense(14400,3)) |> f64 # aggiugnere conv per diminuire numero di parametri
# model = cpu(model)

# Trying to access all layers


model = Flux.Chain(Conv((1,30), 1 => 8), machine, flatten, Dense(536,128), Dense(128,32), Dense(32,2)) |> f64;

model = cpu(model)

# Parameters
params = Flux.params(model)

optimiser = ADAM(0.05)

loss(x,y) = logitcrossentropy(model(x), y)

# Training and plotting
epochs = Int64[]
loss_on_train = Float64[]
loss_on_test = Float64[]
acc_train = Float64[]
acc_test = Float64[]
best_params = Float32[]

for epoch in 1:500

    # Train
    Flux.train!(loss, params, train_data, optimiser)

    
    # Saving losses and accuracies for visualization
    push!(epochs, epoch)
    push!(loss_on_train, loss(x_train, y_train))
    push!(loss_on_test, loss(x_test, y_test))
    push!(acc_train, accuracy(y_train, model(x_train)))
    push!(acc_test, accuracy(y_test, model(x_test)))
    @show loss(x_train, y_train)
    @show loss(x_test, y_test)

    # Saving the best parameters
    if epoch > 1
        if is_best(loss_on_test[epoch-1], loss_on_test[epoch])
            best_params = params
        end
    end
end

@show minimum(loss_on_test)
@show maximum(acc_test)

# Extract and add new trained parameters
if isempty(best_params)
    best_params = params
end

Flux.loadparams!(model, best_params);


# Visualization
plot(epochs, loss_on_train, lab="Training", c=:black, lw=2, ylims = (0,1));
plot!(epochs, loss_on_test, lab="Test", c=:green, lw=2, ylims = (0,1));
title!("Ischemie - conv machine");
yaxis!("Loss");
xaxis!("Training epoch");
savefig("ischemie/visualization/losses/ischemie_conv_loss.png");

plot(epochs, acc_train, lab="Training", c=:black, lw=2, ylims = (0,1));
plot!(epochs, acc_test, lab="Test", c=:green, lw=2, ylims = (0,1));
title!("Ischemie - conv machine");
yaxis!("Accuracy");
xaxis!("Training epoch");
savefig("ischemie/visualization/accuracies/ischemie_conv_accuracy.png");


loss() = logitcrossentropy(model(x_train), y_train);

params = Flux.params(model);


lossfun, gradfun, fg!, p0 = optfuns(loss, params)
res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=500, store_trace=true))

best_params_PM = res.minimizer
#copy flattened optimized params 
copy!(params, best_params_PM)

Flux.loadparams!(model, params)

accuracy(y_test, model(x_test))