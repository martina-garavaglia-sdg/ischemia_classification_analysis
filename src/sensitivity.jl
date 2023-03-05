using ParametricMachinesDemos


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