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
# using VegaLite, DataFrames
# using CairoMakie


# Loading and processing data
train = readdlm("data/ECG200_TRAIN.txt")
test = readdlm("data/ECG200_TEST.txt")

y_train = train[:, 1]
y_test = test[:,1]
y_train = onehotbatch(y_train, (-1,1))
y_test = onehotbatch(y_test, (-1,1))

x_train = permutedims(train[:, 2:end], (2,1))
x_test = permutedims(test[:, 2:end], (2,1))

dtrain = fit(ZScoreTransform, x_train; dims = 1)
StatsBase.transform!(dtrain, x_train)
dtest = fit(ZScoreTransform, x_test; dims = 1)
StatsBase.transform!(dtest, x_test)

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
    device)

@info "Max accuracy on test:" maximum(acc_test)

@save "mymodel.bson" best_model
#using Flux: flatten
machine = machine_type(dimensions, sigmoid; pad=pad, timeblock=timeblock)
model1 = Flux.Chain(embedder, machine, Conv((1,), sum(dimensions) => 2), flatten, Dense(192,2), softmax)
@load "mymodel.bson" model1

# Visualization
Plots.plot(1:n_epochs, loss_on_train, xtickfontsize=10, ytickfontsize=10, xguidefontsize=10, yguidefontsize=10, legendfontsize=12, lw=2, ylim=(0,1), lab="Loss on train")
yaxis!("Loss");
xaxis!("Training epochs");
savefig("visualization/losses/ischemie_rec_loss_reg.png");

plot(1:n_epochs, acc_train, xtickfontsize=10, ytickfontsize=10, xguidefontsize=10, yguidefontsize=10, legendfontsize=12, ylim=(0,1), lw=2, lab="Accuracy on train")
plot!(1:n_epochs, acc_test, xtickfontsize=10, ytickfontsize=10, xguidefontsize=10, yguidefontsize=10, legendfontsize=12, ylim=(0,1), lw=2, lab="Accuracy on test")
yaxis!("Accuracies");
xaxis!("Training epochs");
savefig("visualization/accuracies/ischemie_rec_accuracy_reg.png");

########################################################################
############################# EXPLAINABILITY ###########################
########################################################################

trained_machine = best_model[2]
trained_embedder = best_model[1]
sensitivity, y, z = compute_sensitivity(trained_embedder, trained_machine, x_train)
sensitivity_test, y, z = compute_sensitivity(trained_embedder, trained_machine, x_test)
# test[:,1] .== 1.0
sensitivity_normal = transpose(sensitivity[:,:, 97])
sensitivity_abnormal = transpose(sensitivity[:, :, 7])
serie_normal = x_train[:,:,97]
serie_abnormal = x_train[:,:,7]
plot1 = heatmap(sensitivity_normal, color=:thermal, legend=false, ylab="depth")
savefig("visualization/explainability/exp1.png")
plot2 = heatmap(sensitivity_abnormal, color=:thermal, ylab="depth")
savefig("visualization/explainability/exp2.png")
plot3 = Plots.plot(serie_normal, legend = false, xlab="observations (normal)")
savefig("visualization/explainability/exp3.png")
plot4 = Plots.plot(serie_abnormal, legend = false, xlab="observations (ischemia)")
savefig("visualization/explainability/exp4.png")

Plots.plot(plot1, plot2, plot3, plot4)
savefig("visualization/explainability/explainability_recc.png")



# plot_y_normal = heatmap(transpose(y[:,:,97]), color=:thermal, xlab="y normal")
# plot_y_abnormal = heatmap(transpose(y[:,:,6]), color=:thermal, xlab="y abnormal")

# plot_z_normal = heatmap(transpose(z[:,:,97]), color=:thermal, xlab="z normal")
# plot_z_abnormal = heatmap(transpose(z[:,:,6]), color=:thermal, xlab="z abnormal")

# plot(plot_y_normal,plot_z_normal)



## explore y and z
data = train
data = vcat(data, test)
train_normal = train[train[:,1] .== -1.0,:]
test_normal = test[test[:,1] .== -1.0,:]
data_normal = vcat(train_normal, test_normal)

train_abnormal = train[train[:,1] .== 1.0,:]
test_abnormal = test[test[:,1] .== 1.0,:]
data_abnormal = vcat(train_abnormal, test_abnormal)

data_normal = permutedims(data_normal[:, 2:end], (2,1))
data_abnormal = permutedims(data_abnormal[:, 2:end], (2,1))

data_normal = Flux.unsqueeze(data_normal, 2)
data_abnormal = Flux.unsqueeze(data_abnormal, 2)

output_normal = compute_output_machine(embedder, machine_type(dimensions, sigmoid; pad=pad, timeblock=timeblock), data_normal)
output_ischemia = compute_output_machine(embedder, machine_type(dimensions, sigmoid; pad=pad, timeblock=timeblock), data_abnormal)

mean_healthy = mean(output_normal, dims=3)
var_healthy = var(output_normal, dims=3)
mean_ischemia = mean(output_ischemia, dims=3)
var_ischemia = var(output_ischemia, dims=3)


diff_sensitivity_post_machine = @. (mean_healthy - mean_ischemia) / (var_healthy + var_ischemia)
heatmap(transpose(diff_sensitivity_post_machine[:,:,1])) # quanto un pixel influisce


### umap and mapper
# Input must be of size (n_features, n_samples)

input_embedding = reshape(sensitivity, (96*96,100))
test_embedding = reshape(sensitivity_test[:,:,1:1], (96*96,1))
# UMAP -> dimensionality reduction

gt_train = onecold(y_train) #.== 0
gt_train_normal_inds = findall(gt_train .== 1)

gt_train_ischemia_inds = findall(gt_train .== 2)


# Compute loss on train
loss_on_train_samples = []

for (i,y) in enumerate(eachcol(y_train)) 
    x = x_train[:,:,i:i]
    #x = Flux.unsqueeze(x, dims=3)
    push!(loss_on_train_samples, crossentropy(best_model(x), y))
end


# Fare legenda a parte: triangoli = normal, pallini=ischemia
Gadfly.plot(
    layer(x = embedding[1,gt_train_normal_inds], y = embedding[2,gt_train_normal_inds], Theme(panel_fill="white"), color=loss_on_train[gt_train_normal_inds], shape=[utriangle], Geom.point),
    layer(x = embedding[1,gt_train_ischemia_inds], y = embedding[2,gt_train_ischemia_inds], Theme(panel_fill="white"), color=loss_on_train[gt_train_ischemia_inds], Geom.point))



# Mapper -> cardinality reduction
filter = (data) -> vec(mapslices(p->p[1], data, dims=1))

# Mapper attributes:
#     adj::AbstractMatrix{<:Integer}
#     filter::Vector{<:Real}
#     patches::Vector{Vector{<:Integer}}
#     centers::Matrix{<:Real}


embedding = dimensionality_reduction(input_embedding, test_embedding)


mpr, plt = plot_mapper(embedding, filter) #, loss_on_train_samples
plt

savefig("grafo_test.png")


# gioco con il modello
relative_number_of_ischemia = []

for i in 1:size(mpr.patches)[1]
    n_ischemia = sum(test[mpr.patches[i],1] .== 1)
    #n_normal = length(mpr.patches[i]) - n_ischemia
    push!(relative_number_of_ischemia, n_ischemia/length(mpr.patches[i]))
end

print(string.(relative_number_of_ischemia))