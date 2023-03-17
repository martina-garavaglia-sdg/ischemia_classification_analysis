using Flux
using Flux: onehotbatch, crossentropy, onecold, flatten
using Plots
using ParametricMachinesDemos
using DelimitedFiles
using StatsBase
using Gadfly
using BSON: @load, @save
using Tracker
using ischemia_classification_analysis

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



# Saving model parameters
# weights = Tracker.data.(Flux.params(best_model));
# @save "mymodel.bson" weights

#Loading model parameters
# Model
dimensions = [16,16,16,16,16,16]
timeblock = 16 
pad = 1 
embedder = Conv((1,), 1 => 16)
machine = RecurMachine(dimensions, sigmoid; pad=pad, timeblock=timeblock)
best_model = Flux.Chain(embedder, machine, Conv((1,), sum(dimensions) => 2), flatten, Dense(192,2), softmax) |> cpu

weights = Flux.params(best_model)

@load "mymodel.bson" weights
Flux.loadparams!(best_model, weights)

########################################################################
############################# EXPLAINABILITY ###########################
########################################################################

# Trained machine
trained_machine = best_model[2]
trained_embedder = best_model[1]

# Sensitivity matrix on train and test
sensitivity, y, z = compute_sensitivity(trained_embedder, trained_machine, x_train)
sensitivity_test, y, z = compute_sensitivity(trained_embedder, trained_machine, x_test)

# Exploring sensitivity for a normal heartbeat and an ischemia one
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


####################################################################
################### UMAP - dimensionality reduction #################
#####################################################################

# Input must be of size (n_features, n_samples)
input_embedding = reshape(sensitivity, (96*96,100))
test_embedding = reshape(sensitivity_test[:,:,1:1], (96*96,1))

# Ground truth
gt_train = onecold(y_train) #.== 0
gt_train_normal_inds = findall(gt_train .== 1)
gt_train_ischemia_inds = findall(gt_train .== 2)


# Compute loss on train -> for graph representation
loss_on_train_samples = []

for (i,y) in enumerate(eachcol(y_train)) 
    x = x_train[:,:,i:i]
    push!(loss_on_train_samples, crossentropy(best_model(x), y))
end


# Performing UMAP
embedding = dimensionality_reduction(input_embedding, input_embedding)


# Fare legenda a parte: triangoli = normal, pallini=ischemia
Gadfly.plot(
    layer(x = embedding[1,gt_train_normal_inds], y = embedding[2,gt_train_normal_inds], Theme(panel_fill="white"), color=loss_on_train[gt_train_normal_inds], shape=[utriangle], Geom.point),
    layer(x = embedding[1,gt_train_ischemia_inds], y = embedding[2,gt_train_ischemia_inds], Theme(panel_fill="white"), color=loss_on_train[gt_train_ischemia_inds], Geom.point))


####################################################################
################### MAPPER -> cardinality reduction ################
####################################################################

# Filter function
filter = (data) -> vec(mapslices(p->p[1], data, dims=1))

# Mapper attributes:
#     adj::AbstractMatrix{<:Integer}
#     filter::Vector{<:Real}
#     patches::Vector{Vector{<:Integer}}
#     centers::Matrix{<:Real}

# Mapper
mpr, plt = plot_mapper(embedding, filter) #, loss_on_train_samples
plt


# Brutto modo per inserire nel grafo in TRAIN la quantità relativa di osservazioni di ischemia
# relative_number_of_ischemia = []

# for i in 1:size(mpr.patches)[1]
#     n_ischemia = sum(test[mpr.patches[i],1] .== 1)
#     push!(relative_number_of_ischemia, n_ischemia/length(mpr.patches[i]))
# end

# print(string.(relative_number_of_ischemia))