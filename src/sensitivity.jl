using ParametricMachinesDemos
using UMAP
using Plots
using TDA
using Makie
using GLMakie


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
    μ = []
    #σ = []
    for i in 1:size(mpr.patches)[1]
        push!(μ, mean(losses[mpr.patches[i]]))
        #push!(σ, std(losses[mpr.patches[i]]))
    end
    return μ#, σ
end


function n_values_in_cluster(mpr)
    values = []
    for i in 1:size(mpr.patches)[1]
        push!(values, size(mpr.patches[i])[1])
    end
    return values
end


function dimensionality_reduction(data_train, data_test; n_components = 40) # input -> flatten della sensitivity
    model = UMAP_(data_train, n_components)
    embedding = transform(model, data_test)
    return embedding
end

# cardinality reduction
function plot_mapper(data, filter, losses=0)
    mpr = TDA.mapper(data, filter=filter, seed=0, intervals=4, overlap=0.7)
    mean_losses = mean_losses_from_mpr(mpr, losses)
    
    return mpr, Plots.plot(mpr; complex_layout=TDA.constant_layout, xlims=(-5,5), ylims=(-5,5), color=:Blues_9, zcolor=mean_losses, xticks=:none, yticks=:none)
end


function sensitivity_cam(s_map, ts)
    s1 = s_map[1,:]#mean(s_map[1:16,:], dims= 1)
    Makie.scatter(1:size(ts)[1], vec(ts), color=s1, colormap=:thermal)
end