using Flux.Data: DataLoader
using Flux: train!, flatten, crossentropy
using Flux
using Optim, FluxOptTools
using Random
using LineSearches
using ParametricMachinesDemos


function train_forecast(x_train::AbstractArray, y_train::Flux.OneHotArray, x_test::AbstractArray, y_test::Flux.OneHotArray,
    machine_type,
    dimensions, timeblock = 1, pad = 1, opt = "Adam", learning_rate = 0.01, line_search = BackTracking(),
    n_epochs=100, device=cpu)

    Random.seed!(3)

    if machine_type == DenseMachine
        machine = machine_type(dimensions, sigmoid);
        model = Flux.Chain(machine, Dense(sum(dimensions), 2), softmax) |> device
    end

    if machine_type == RecurMachine
        machine = machine_type(dimensions, sigmoid; pad=pad, timeblock=timeblock)
        model = Flux.Chain(machine, Conv((1,), sum(dimensions) => 2), flatten, Dense(192,2), softmax) |> device
    end


    params = Flux.params(model)

    if opt == "Adam"
        return train_ADAM(x_train, y_train, x_test, y_test, learning_rate, params, model, n_epochs)
    end

    if opt == "LBFGS"
        return train_LBFGS(x_train, y_train, x_test, y_test, model, line_search, params, n_epochs)
    end

    if opt == "ConjugateGradient"
        return train_ConjGrad(x_train, y_train, x_test, y_test, model, line_search, params, n_epochs)
    end

    @error "This optimizer has not been implemented yet."
end



function train_ADAM(x_train::AbstractArray, y_train::Flux.OneHotArray, 
                    x_test::AbstractArray, y_test::Flux.OneHotArray, 
                    learning_rate, params, model, n_epochs)

    train_data = DataLoader((x_train, y_train); batchsize = 32, shuffle = true);
    opt = ADAM(learning_rate);

    loss(x,y) = crossentropy(model(x), y)
    
    @info "Starting training."
    loss_on_train = Float64[]
    acc_train = Float64[]
    acc_test = Float64[]
    best_params = Float32[]

    for epoch in 1:n_epochs

        # Train
        Flux.train!(loss, params, train_data, opt)

        # Saving losses and accuracies for visualization
        push!(loss_on_train, loss(x_train, y_train))
        push!(acc_train, accuracy(y_train, model(x_train)))
        push!(acc_test, accuracy(y_test, model(x_test)))
        @show loss(x_train, y_train)

        # Saving the best parameters
        if epoch > 1
            if is_best(loss_on_train[epoch-1], loss_on_train[epoch])
                best_params = params
            end
        end
    end

    if isempty(best_params)
        best_params = params
    end

    Flux.loadparams!(model, best_params);

    return best_params, model, loss_on_train, acc_train, acc_test

end


function train_LBFGS(x_train::AbstractArray, y_train::Flux.OneHotArray, 
                        x_test::AbstractArray, y_test::Flux.OneHotArray, 
                        model, line_search, params, n_epochs)

    @info "Starting training."
    loss() = crossentropy(model(x_train), y_train);
   
    _, _, fg!, p0 = optfuns(loss, params)
    res = Optim.optimize(Optim.only_fg!(fg!), p0, LBFGS(linesearch = line_search), Optim.Options(iterations=n_epochs, store_trace=true))

    best_params = res.minimizer

    copy!(params, best_params)

    Flux.loadparams!(model, params)

    acc_train = accuracy(y_train, model(x_train))
    acc_test = accuracy(y_test, model(x_test))
    loss_on_train = crossentropy(model(x_train), y_train)

    return params, model, loss_on_train, acc_train, acc_test

end


function train_ConjGrad(x_train::AbstractArray, y_train::Flux.OneHotArray, 
                        x_test::AbstractArray, y_test::Flux.OneHotArray, 
                        model, line_search, params, n_epochs)

    @info "Starting training."
    loss() = crossentropy(model(x_train), y_train);
    lossfun, gradfun, fg!, p0 = optfuns(loss, params)
    res = Optim.optimize(Optim.only_fg!(fg!), p0, ConjugateGradient(linesearch = line_search), Optim.Options(iterations=n_epochs, store_trace=true))

    best_params = res.minimizer

    copy!(params, best_params)

    Flux.loadparams!(model, params)

    acc_train = accuracy(y_train, model(x_train))
    acc_test = accuracy(y_test, model(x_test))
    loss_on_train = crossentropy(y_train, model(x_train))

    return params, model, loss_on_train, acc_train, acc_test

end