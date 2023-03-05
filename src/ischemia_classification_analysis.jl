module ischemia_classification_analysis

using Statistics: mean, std
using Flux 
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten, params, loadparams!, crossentropy
using Flux.Data: DataLoader
using Random

export accuracy, is_best, train_forecast, train_ADAM, train_LBFGS, train_ConjGrad, compute_sensitivity

include("metrics.jl")
include("train.jl")
include("sensitivity.jl")

end
