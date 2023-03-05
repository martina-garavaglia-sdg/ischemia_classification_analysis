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
embedder = Dense(96,64)  
dimensions = [64,32,16,8]
machine = DenseMachine(dimensions, sigmoid);
model = Flux.Chain(embedder, machine)



# implementazione di compute_sensitivity che usa internals del pacchetto (va pacchettizzato in futuro)
function compute_sensitivity(m, x)
    filtrations = ParametricMachinesDemos.filtrations(m, x)
    y, z = ParametricMachinesDemos.solve(x, nothing, m.W, m.σ, filtrations) # prima e dopo nonlin
    return ParametricMachinesDemos.derivative.(z, m.σ, y) # derivata di sigma su y
end

# restituisce sensitivity dello stato globale della macchina parametrica

function compute_sensitivity(e, m, x)
    input_machine = e(x)
    compute_sensitivity(m, input_machine)
end


#input_machine = embedder(x_train)

sensitivity = compute_sensitivity(machine, input_machine)


heatmap(sensitivity, color=:thermal)