module ischemia_classification_analysis

using Statistics: mean, std
using Flux 
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten, params, loadparams!, crossentropy
using Flux.Data: DataLoader
using Random
using ComputationalHomology
using RecipesBase

export accuracy, is_best, train_classification, train_ADAM, train_LBFGS, train_ConjGrad, compute_sensitivity, compute_output_machine, dimensionality_reduction, plot_mapper, sensitivity_cam

include("metrics.jl")
include("train.jl")
include("sensitivity.jl")


end
