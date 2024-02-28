# Implementation of a small neural network for three patients of the Eleveld data set, with fastpksim simulation

using Pkg
cd(@__DIR__)
Pkg.activate(".")

using Flux, Plots, Random, StatsBase

seed = 12345
Random.seed!(seed) # for reproducibility

## Read Eleveld data
include("fastpksim.jl")
include("get_elevelddata.jl")
include("fcts.jl")

# Train the network
function trainNN(n_epochs, seed = 1234 ;n_pat = 1031, starting_pat = 1, verbose=true)
    x, y, _ = get_data(starting_pat, n_pat) # get eleveld data

    Random.seed!(seed) # for reproducibility
    nn = createNN() # Create the network

    opt = ADAM(5e-4) # Define the optimizer
    @time nn, losses, losses_mdales = train!(nn, x, y, n_epochs, opt, loss=loss_ALE; verbose) # Train the network

    print("Loss after training: ", loss_ALE(nn, x, y)) # loss after training

    return losses, losses_mdales, nn, x, y
end

# losses, losses_mdales, nn, x, y = trainNN(5000, 12345,verbose=true)
losses, losses_mdales, nn, x, y = trainNN(5000, 12345,verbose=false)

# Test
# losses, losses_mdales, nn, x, y = trainNN(2, 12345, verbose=false)


## Plot the results
# p = plot(log.([losses1; losses2]), label="Logarithmic training loss", xlabel="Epoch", ylabel="Loss")
# display(p)

# t_obs,y_pred = get_predictions_fastpksim(nn,x)
# # # 1000 epochs, 1e-4 learning rate, full data set takes 2770 seconds

# # ## Plot predictions vs observations

# p = scatter(y, y_pred, label="", xlabel="Observed", ylabel="Predicted", legend=:bottomright)
# display(p)


