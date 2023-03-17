using ParametricMachinesDemos
using UMAP
using Plots
using TDA


function compute_sensitivity(m, x)
    filtrations = ParametricMachinesDemos.filtrations(m, x)
    y, z = ParametricMachinesDemos.solve(x, nothing, m.W, m.σ, filtrations) # prima e dopo nonlin
    return ParametricMachinesDemos.derivative.(z, m.σ, y), y, z # derivata di sigma su y
end
# y prima della linearità, z dopo linearità

# restituisce sensitivity dello stato globale della macchina parametrica

function compute_sensitivity(e, m, x)
    input_machine = e(x)
    compute_sensitivity(m, input_machine)
end


function compute_output_machine(e, m, input)
    return m(e(input))
end



function mean_losses_from_mpr(mpr, losses)
    means = []
    for i in 1:size(mpr.patches)[1]
        push!(means, mean(losses[mpr.patches[i]]))
    end
    return means
end


function n_values_in_cluster(mpr)
    values = []
    for i in 1:size(mpr.patches)[1]
        push!(values, size(mpr.patches[i])[1])
    end
    return values
end


function dimensionality_reduction(data_train, data_test; n_components = 20) # input -> flatten della sensitivity
    model = UMAP_(data_train, n_components)
    embedding = transform(model, data_test)
    return embedding
end

# cardinality reduction
function plot_mapper(data, filter, losses=0)
    mpr = TDA.mapper(data, filter=filter, seed=0, intervals=5, overlap=0.6)
    #mean_losses = mean_losses_from_mpr(mpr, losses)
    
    return mpr, Plots.plot(mpr; complex_layout=TDA.constant_layout, color=:Blues_3,
                                  xlims=(-2,4), ylims=(-5,3)) #color=:Blues_9, zcolor=mean_losses,
end


