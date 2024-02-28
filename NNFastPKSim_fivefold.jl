# Implementation of a small neural network for three patients of the Eleveld data set, with fastpksim simulation
# currently, using MSE loss

using Pkg
cd(@__DIR__)
Pkg.activate(".")

using Flux, Plots, Random
# using DifferentialEquations, SciMLSensitivity

seed = 12345
Random.seed!(seed) # for reproducibility

## Read Eleveld data
include("fastpksim.jl")
include("get_elevelddata.jl")
include("fcts.jl")


## Five fold cross-validation
function train_fivefold(n_epochs, seed=1234; n_pat=1031, starting_pat=1, verbose=true)
    Random.seed!(seed) # for reproducibility
    x, y, _ = get_data(starting_pat, n_pat) # get eleveld data

    n_pat = length(x)
    percentfold = 0.8
    shuffled_indices = randperm(n_pat)
    n_pat_eachtest = Int(ceil(n_pat * (1 - percentfold)))

    # n_epochs = 100
    totaldiffloss = 0

    @time Threads.@threads for fold = 1:5
        test_ind_low = (fold - 1) * n_pat_eachtest + 1
        test_ind_high = test_ind_low + n_pat_eachtest - 1

        if test_ind_high > n_pat
            test_ind_high = n_pat
        end

        test_indices = shuffled_indices[test_ind_low:test_ind_high]
        train_indices = filter(x -> !(x in test_indices), shuffled_indices)

        xtrain = x[train_indices]
        ytrain = y[train_indices]
        xtest = x[test_indices]
        ytest = y[test_indices]

        nn = createNN()

        print("Average loss per person before training on training set $fold: ", round(loss_ALE(nn, xtrain, ytrain) / length(train_indices), digits=6), "\n") # loss after training

        opt = ADAM(5e-4) # Define the optimizer
        nn, losses, losses_mdales = train!(nn, xtrain, ytrain, n_epochs, opt, loss=loss_ALE; verbose) # Train the network

        # opt = ADAM(1e-4)
        # nn, losses = train!(nn, xtrain, ytrain, n_epochs, opt, loss=loss_ALE, verbose=false)

        # print("Average loss per person after training on training set $fold: ", round(loss_ALE(nn, xtrain, ytrain)/length(train_indices), digits=6), "\n") # loss after training
        # print("Average loss per person after training on test set $fold: ", round(loss_ALE(nn, xtest, ytest)/length(test_indices), digits=6), "\n") # loss after training

        if verbose
            loss_mdale_train = 0.0
            for i in eachindex(ytrain)
                _, yp = get_predictions_fastpksim(nn, [xtrain[i]])
                loss_mdale_train += MdALE(ytrain[i], yp[1])
            end
            loss_mdale_train /= length(ytrain)

            loss_mdale_test = 0.0
            for i in eachindex(ytest)
                _, yp = get_predictions_fastpksim(nn, [xtest[i]])
                loss_mdale_test += MdALE(ytest[i], yp[1])
            end
            loss_mdale_test /= length(ytest)


            print("Average loss per person after training on training set $fold: ", round(loss_mdale_train, digits=6), "\n") # loss after training
            print("Average loss per person after training on test set $fold: ", round(loss_mdale_test, digits=6), "\n") # loss after training

        end
        # diffloss = loss_ALE(nn, xtest, ytest) / length(test_indices) - loss_ALE(nn, xtrain, ytrain) / length(train_indices)
        # totaldiffloss += diffloss
        # print("Difference between test and training loss: ", round(diffloss, digits=6), "\n") # loss after training

    end
    return losses_mdales[end]
end

final_mdale_loss = train_fivefold(5000, 12345, verbose=true)

# 
# totaldiffloss = train_fivefold(1,x,y,1234)

# print("Loss after training: ", loss_ALE(nn, x, y)) # loss after training

# plot(log.(losses), label="Logarithmic training loss", xlabel="Epoch", ylabel="Loss")

# ## Plot the results
# t_obs, y_pred = get_predictions_fastpksim(nn, x)
# # 1000 epochs, 1e-4 learning rate, full data set takes 2770 seconds

# ## Plot predictions vs observations
# scatter(y, y_pred, label="", xlabel="Observed", ylabel="Predicted", legend=:bottomright)