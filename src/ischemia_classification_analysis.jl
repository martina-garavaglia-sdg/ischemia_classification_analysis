module ischemia_classification_analysis

using Statistics: mean, std
using Flux 
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten, params, loadparams!, crossentropy
using Flux.Data: DataLoader
using Random

export accuracy, is_best, train_forecast

include("metrics.jl")
include("train.jl")

end
