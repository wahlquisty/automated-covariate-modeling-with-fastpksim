using StatsBase

function get_data(first_pat, n_pat)
    # Read data
    x_all, y_all = get_elevelddata()
    Ω, y_eleveld, _ = get_predictions_eleveld()

    # Get training data
    x = x_all[first_pat:n_pat+first_pat-1]
    y = y_all[first_pat:n_pat+first_pat-1]
    return x, y, y_eleveld
end


"""
     loss_MSE(nn::CHain, x::InputData, y::Vector)

Computes the error over a Flux model given x and y data (one patiente) (by calling the function loss_MSE(model,x,y))
Output from network model results in a vector of covariates, which inputs to a simulation together with simulation data (input, time) from xdata object. Mse is computed between predicted output concentrations (at time (x.time)[x.youts]) and measured concentrations y.

# Arguments:
- `nn`: Flux model (Chain)
- `x`: InputData struct object. x training data.
- `y`: Vector. y training data, measurements at instances xdata.youts.

Returns the total loss over all training data.
"""
function loss_MSE(nn::Chain, x::InputData, y::Vector)
    u = x.u # Infusion rates
    v = x.v # Bolus doses
    hs = x.hs # Time differences between rate changes. Δ(t+1)-Δ(t) = hs

    ϕhat_scaled = get_modeloutput(nn, x.covariates) # Vector with predicted PK parameters (k10, k12 etc) scaled using x.normalization
    ϕhat = ϕhat_scaled .* x.normalization # Scale predicted PK parameters from (0,1) to range (0,maxval in dataset)

    V1inv, λ, λinv, R = PK3(ϕhat) # Create necessary matrices for simulation of 3rd order compartment model

    totalloss = 0.0 # squared error
    j = 1 # counter to keep track of next free spot in y
    x_state = zeros(eltype(u), 3) # initial state
    for i in eachindex(hs) # Iterate through all time samples
        if i in x.youts # if we want to compute output
            x_state, yi = @inbounds updatestateoutput(x_state, hs[i], V1inv, λ, λinv, R, u[i], v[i]) # update state and compute output
            totalloss += compute_squarederror(y[j], yi) # only compute loss when we have observations
            j += 1
        else
            x_state = @inbounds updatestate(x_state, hs[i], λ, λinv, u[i], v[i]) # update state
        end
    end
    return totalloss / length(x.youts) # Mean squared error
end

compute_squarederror(y, yhat) = abs2(y - yhat)

"""
     loss_ALE(nn::Chain, x::InputData, y::Vector)

Computes the ALE (absolute logarithmic error) over a Flux model given x and y data (one patient).
Output from network model results in a vector of covariates, which inputs to a simulation together with simulation data (input, time) from x object. ALE is computed between predicted output concentrations (at time (x.time)[x.youts]) and measured concentrations y.

# Arguments:
- `nn`: Flux model of parallel neural networks using Split.
- `x`: InputData struct object. x training data.
- `y`: Vector. y training data, measurements at instances x.youts.

Returns the ALE loss over all x and y at instances (x.time)[x.youts].
"""
function loss_ALE(nn::Chain, x::InputData, y::Vector) # compute mse loss
    u = x.u # Infusion rates
    v = x.v # Bolus doses
    hs = x.hs # Time differences between rate changes. Δ(t+1)-Δ(t) = hs

    ϕhat_scaled = get_modeloutput(nn, x.covariates) # Vector with predicted PK parameters (k10, k12 etc) scaled using x.normalization
    ϕhat = ϕhat_scaled .* x.normalization # Scale predicted PK parameters from (0,1) to range (0,maxval in dataset)

    V1inv, λ, λinv, R = PK3(ϕhat) # Create necessary matrices for simulation of 3rd order compartment model

    totalloss = 0.0 # squared error
    j = 1 # counter to keep track of next free spot in y
    x_state = zeros(eltype(u), 3) # initial state
    for i in eachindex(hs) # Iterate through all time samples
        if i in x.youts # if we want to compute output
            x_state, yi = @inbounds updatestateoutput(x_state, hs[i], V1inv, λ, λinv, R, u[i], v[i]) # update state and compute output
            if yi > 0 # If predicted concentration is zero, do not add to total loss
                # if !isapprox(0, yi,atol=1e-8)
                # totalloss += abs(yi - y[j]) # TEMPORARY!

                totalloss += compute_ALE(y[j], yi) # Compute ALE
                # elseif isnan(yi)
                #    return NaN
            end
            j += 1
        else
            x_state = @inbounds updatestate(x_state, hs[i], λ, λinv, u[i], v[i]) # update state
        end
    end
    return totalloss / length(y)
end

compute_ALE(y, yhat) = abs(log(y / abs(yhat)))

function get_modeloutput(nn::Chain, x)
    return vec(nn(x))
end


"""
     loss(nn:Chain, x::InputData, y)

Computes the loss over a Flux model given x and y data (many patients).
Output from network model results in a vector of covariates, which inputs to a simulation together with simulation data (input, time) from x object. Loss is computed between predicted output concentrations (at time (x.time)[x.youts]) and measured concentrations y.

# Arguments:
- `nn`: Flux model (Chain)
- `x`: Vector{InputData} struct objects. x training data.
- `y`: Vector{Vector}. y training data, measurements at instances x.youts.

Returns the mse loss over all x and y at instances (x.time)[x.youts].
"""
function loss_ALE(nn::Chain, x::Vector{InputData}, y)
    totalloss = 0.0
    for i in eachindex(x)
        totalloss += loss_ALE(nn, x[i], y[i]) # Compute loss for each patient
    end
    return totalloss
end

"""
     train!(model, x, y, n_epochs, opt, ps)

Trains the Flux model for n_epochs given parameters ps and optimizer opt on training data x and y. Computes the gradient of the loss function loss(model,x,y) and updates the model based on the choice of optimizer.

# Arguments:
- `model`: Flux model of parallel neural networks using Split.
- `x`: Vector{InputData} struct objects. x training data.
- `y`: Vector{Vector}. y training data.
- `n_epochs`: Number of training epochs.
- `opt`: Optimizer object. For example Flux.Adam(0.005).
- `ps`: Params object. Model parameters from Flux.params(model).

Returns the updated model and training losses at each epoch.
"""
function train!(nn::Chain, x, y, n_epochs, opt; loss=loss_ALE, verbose=true)
    ps = Flux.params(nn)
    n_pat = Float32(length(y))
    λ = 1.0f-1 / n_pat#1f-4 # L2 regularisation parameter

    if verbose
        print("Loss before training: ", round(loss(nn, x, y), digits=6), "\n") # loss before training
    end
    losses = zeros(n_epochs)
    loss_mdales = zeros(n_epochs)

    for epoch = 1:n_epochs
        if verbose
            if iszero(epoch % 100)
                print("Training epoch: ", epoch, "\n")
            end
        end
        loss_epoch = 0.0
        loss_mdale_epoch = 0.0

        for i in eachindex(y)
            loss_i, grads = Flux.withgradient(ps) do
                loss(nn, x[i], y[i]) + λ * sum(pen_l2, ps)  # added l2 regularisation
            end
            Flux.update!(opt, ps, grads)
            loss_epoch += loss_i
        end

        if verbose # compute mdale
            for i in eachindex(y)
                _, yp = get_predictions_fastpksim(nn, [x[i]])
                loss_mdale_i = MdALE(y[i], yp[1])
                loss_mdale_epoch += loss_mdale_i
            end
        end
        
        losses[epoch] = loss_epoch
        loss_mdales[epoch] = loss_mdale_epoch
    end
    if verbose
        print("Loss after training: ", round(loss(nn, x, y), digits=6), "\n") # loss after training
    end

    loss_mdales./=n_pat
    return nn, losses, loss_mdales
end

pen_l2(x::AbstractArray) = sum(abs2, x) / 2

# Define the network
function createNN(neurons=64)
    nn = Chain(
        Dense(5, neurons, relu), # Input/hidden layer
        Dense(neurons, neurons, relu), # Hidden layer
        Dense(neurons, 6, sigmoid)) # Output/hidden layer
    return nn
end

function get_predictions_fastpksim(nn, x)
    n_pat = length(x)
    y_pred = Vector{Vector{Float32}}(undef, n_pat)
    t_obs = Vector{Vector{Float32}}(undef, n_pat)

    for i in eachindex(x)
        θhat_scaled = get_modeloutput(nn, x[i].covariates) # Vector with predicted PK parameters (k10, k12 etc) scaled using x. normalization
        θhat = θhat_scaled .* x[i].normalization # Scale predicted PK parameters from (0,1) to range (0,maxval in dataset)
        y_pred[i] = pk3sim(θhat, x[i].u, x[i].v, x[i].hs, x[i].youts)

        t_obs[i] = x[i].time[x[i].youts]
    end
    return t_obs, y_pred
end

function MdALE(y_meas, y_pred) # Median Absolute Logarithmic Error: median(abs(log(C_observed/C_predicted)))
    mdale = 0.0
    ind = findall(x -> x = !isapprox(x, 0.0, atol=1e-6), y_pred)
    if length(ind) > 0
        mdale = median(abs.(log.(y_meas[ind] ./ abs.(y_pred[ind]))))
    end
    return mdale
end
function MeanMdALE(y, y_pred)
    mdale = 0.0
    for i in eachindex(y)
        mdale += MdALE(y[i], y_pred[i])
    end
    return mdale / length(y)
end