using DelimitedFiles
using Flux
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten, OneHotMatrix
using Statistics: mean, std
using Flux.Data: DataLoader
using Plots
using ParametricMachinesDemos
using Optim, FluxOptTools
using Random
#using ischemia_classification_analysis


# Accuracy
function accuracy(y_true::OneHotMatrix, y_pred::Any)
    if size(y_true) == size(y_pred)
        a = onecold(y_true, 0:(size(y_true)[1]-1))
        b = onecold(y_pred, 0:(size(y_pred)[1]-1))
        l = size(a)[1]
        return sum(a .== b) / l
    else
        error("Error: true labels and predicted labels must have the same size")
    end
end


# Best model parameters evaluating losses
function is_best(old_loss, new_loss)
    return old_loss > new_loss
end



train = readdlm("data/ECG200_TRAIN.txt")
test = readdlm("data/ECG200_TEST.txt")

y_train = train[:,1]
y_test = test[:,1]
y_train = onehotbatch(y_train, (-1,1))
y_test = onehotbatch(y_test, (-1,1))

x_train = permutedims(train[:, 2:end], (2,1))
x_test = permutedims(test[:, 2:end], (2,1))



# Loading
train_data = DataLoader((x_train, y_train); batchsize = 32, shuffle = true);
test_data = DataLoader((x_test, y_test); batchsize = 32, shuffle = true);

dimensions = [32,16,8];

# Define the parametric machine
machine = DenseMachine(dimensions, sigmoid);

model = Flux.Chain(Dense(96, 32), machine, Dense(sum(dimensions), 2)) |> f64;

model = cpu(model)


# Parameters
Random.seed!(3)
params = Flux.params(model);

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

@show maximum(acc_test)
@show minimum(loss_on_train)
@show minimum(loss_on_test)

# Extract and add new trained parameters
if isempty(best_params)
    best_params = params
end

Flux.loadparams!(model, best_params);


# Visualization
plot(epochs, loss_on_train, lab="Training loss", lw=2, ylims = (0,1));
plot!(epochs, loss_on_test, lab="Test loss", lw=2, ylims = (0,1));
#title!("Ischemia - dense machine");
yaxis!("Losses");
xaxis!("Training epochs");
savefig("visualization/losses/ischemie_dense_loss.png");


plot(epochs, acc_test, lab="Test accuracy", lw=2, ylims = (0,1));
plot!(epochs, acc_train, lab="Test accuracy", lw=2, ylims = (0,1));
#title!("Ischemie - dense machine");
yaxis!("Accuracy");
xaxis!("Training epochs");
savefig("visualization/accuracies/ischemie_dense_accuracy.png");



########
# LBFGS
loss() = logitcrossentropy(model(x_train), y_train);

params = Flux.params(model);


lossfun, gradfun, fg!, p0 = optfuns(loss, params)
res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=500, store_trace=true))

best_params_PM = res.minimizer
#copy flattened optimized params 
copy!(params, best_params_PM)

Flux.loadparams!(model, params)

accuracy(y_test, model(x_test))
logitcrossentropy(y_test, model(x_test))
logitcrossentropy(y_train, model(x_train))


