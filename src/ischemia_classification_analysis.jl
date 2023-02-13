module ischemia_classification_analysis

using Statistics: mean, std
using Flux 
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten, params, loadparams!
using Flux.Data: DataLoader

export accuracy, is_best

include("metrics.jl")

end
