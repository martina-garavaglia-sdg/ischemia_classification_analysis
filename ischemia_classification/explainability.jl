using DelimitedFiles
using Flux
using Flux: onehotbatch, crossentropy
using ischemia_classification_analysis
using ParametricMachinesDemos
using LineSearches
using Flux.Data: DataLoader


# Split train test data
train = readdlm("data/ECG200_TRAIN.txt")
test = readdlm("data/ECG200_TEST.txt")

y_train = train[:,1]
y_test = test[:,1]
y_train = onehotbatch(y_train, (-1,1))
y_test = onehotbatch(y_test, (-1,1))

x_train = permutedims(train[:, 2:end], (2,1))
x_test = permutedims(test[:, 2:end], (2,1))

train_data = DataLoader((x_train, y_train); batchsize = 32, shuffle = true);

# crea esempio di macchina parametrica
dimensions = [16,16,16,16,16,16]
machine = DenseMachine(dimensions, sigmoid);


# implementazione di compute_sensitivity che usa internals del pacchetto (va pacchettizzato in futuro)
function compute_sensitivity(m, x)
    filtrations = ParametricMachinesDemos.filtrations(m, x)
    y, z = ParametricMachinesDemos.solve(x, nothing, m.W, m.σ, filtrations)
    return ParametricMachinesDemos.derivative.(z, m.σ, y)
end

# restituisce sensitivity dello stato globale della macchina parametrica
sensitivity = compute_sensitivity(machine, x_train)


heatmap(sensitivity)