# Plot results from training of nn

using Plots, StatsBase

print("Loss after training: ", loss_ALE(nn, x, y)) # loss after training

# plot(log.(losses), label="Logarithmic training loss", xlabel="Epoch", ylabel="Loss")
p = plot(log.(losses), label="Logarithmic training loss", xlabel="Epoch", ylabel="Loss")
display(p)

## Plot the results
t_obs, y_pred = get_predictions_fastpksim(nn, x)

## Plot predictions vs observations
p = scatter(y, y_pred, label="", xlabel="Observed", ylabel="Predicted", legend=:bottomright)
display(p)


# Compute mean(ALE) over data set
n_pat = length(y)
print("Mean(ALE) over the data set is: ", loss_ALE(nn, x, y) / n_pat)

# Plot predictions vs observations for each patient in log-log scale, ignoring zero entries
ys = Vector{Float32}(undef, 0)
y_preds = Vector{Float32}(undef, 0)
for i in eachindex(x)
    for j in eachindex(y[i])
        if j > 0 && y_pred[i][j] > 0
            ys = append!(ys, y[i][j])
            y_preds = append!(y_preds, y_pred[i][j])
        end
    end
end
p = scatter(ys, y_preds, label="", xlabel="Observed", ylabel="Predicted", legend=:bottomright, xscale=:log10, yscale=:log10, xtick=10 .^ (-4.0:2.0), ytick=10 .^ (-4.0:2.0), xlims=(1e-4, 1e2), ylims=(1e-4, 1e2))
display(p)

# using StatsBase



print("Mean(MdALE) over the data set is: ", MeanMdALE(y, y_pred))


# Plot 3 patients
p1 = plot(t_obs[1], y[1], label="Measured concentration, patient 1", color=1, marker=:circle)
scatter!(p1, t_obs[1], y_pred[1], label="Predicted concentration, patient 1", color=1, shape=:utriangle)
p2 = plot(t_obs[2], y[2], label="Measured concentration, patient 2", color=2, marker=:circle)
scatter!(p2, t_obs[2], y_pred[2], label="Predicted concentration, patient 2", color=2, shape=:utriangle)
p3 = plot(t_obs[3], y[3], label="Measured concentration, patient 3", color=3, marker=:circle)
scatter!(p3, t_obs[3], y_pred[3], label="Predicted concentration, patient 3", color=3, shape=:utriangle)
p = plot(p1, p2, p3, size=(600, 600), layout=grid(3, 1))
display(p)



function MdLE(y_meas, y_pred) # Median Absolute Logarithmic Error: median(abs(log(C_observed/C_predicted)))
    mdle = 0.0
    ind = findall(x -> x = !isapprox(x, 0.0, atol=1e-6), y_pred)
    if length(ind) > 0
        mdle = median(log.(y_meas[ind] ./ abs.(y_pred[ind])))
    end
    return mdle
end
function MeanMdLE(y, y_pred)
    mdle = 0.0
    for i in eachindex(y)
        mdle += MdLE(y[i], y_pred[i])
    end
    return mdle / length(y)
end

function MdPE(y_meas, y_pred) # Prediction Error: abs((C_observed - C_predicted)/C_predicted* 100 %))
    mdpe = 0.0
    ind = findall(x -> x = !isapprox(x, 0.0, atol=1e-6), y_pred)
    if length(ind) > 0
        mdpe = median(100 * (y_meas[ind] .- y_pred[ind]) ./ y_pred[ind]) # %
    end
    return mdpe
end
function MeanMdPE(y, y_pred)
    mdpe = 0.0
    for i in eachindex(y)
        mdpe += MdPE(y[i], y_pred[i])
    end
    return mdpe / length(y)
end

function MdAPE(y_meas, y_pred) # Median Absolute Prediction Error: median(abs((C_observed - C_predicted)/C_predicted* 100 %))
    mdape = 0.0
    ind = findall(x -> x = !isapprox(x, 0.0, atol=1e-6), y_pred)
    if length(ind) > 0
        mdape = 100 * median(abs.((y_meas[ind] .- y_pred[ind]) ./ y_pred[ind])) # %
    end
    return mdape
end

function MeanMdPE(y, y_pred)
    mdpe = 0.0
    for i in eachindex(y)
        mdpe += MdPE(y[i], y_pred[i])
    end
    return mdpe / length(y)
end
function MeanMdAPE(y, y_pred)
    mdape = 0.0
    for i in eachindex(y)
        mdape += MdAPE(y[i], y_pred[i])
    end
    return mdape / length(y)
end

print("Mean(MdALE) over the data set is: ", MeanMdALE(y, y_pred))
print("Mean(MdLE) over the data set is: ", MeanMdLE(y, y_pred))
print("Mean(MdAPE) over the data set is: ", MeanMdAPE(y, y_pred))
print("Mean(MdPE) over the data set is: ", MeanMdPE(y, y_pred))


## Plot spread of V1 and transfer rates
n_pat = length(x)
k10s = Vector{Float32}(undef, n_pat)
k12s = Vector{Float32}(undef, n_pat)
k21s = Vector{Float32}(undef, n_pat)
k13s = Vector{Float32}(undef, n_pat)
k31s = Vector{Float32}(undef, n_pat)
V1s = Vector{Float32}(undef, n_pat)

for i in eachindex(x)
    covariates = x[i].covariates
    pkparams = nn(covariates).*x[i].normalization # k10, k12, k21, k13, k31, V1
    k10s[i] = pkparams[1]
    k12s[i] = pkparams[2]
    k21s[i] = pkparams[3]
    k13s[i] = pkparams[4]
    k31s[i] = pkparams[5]
    V1s[i] = pkparams[6]./1000
end

# p = plot(layout=grid(2, 3))
h1 = histogram(k10s, label="k10", xlabel="k10 [1/s]", ylabel="Frequency", legend=:bottomright)
h2 = histogram(k12s, label="k12", xlabel="k12 [1/s]", ylabel="Frequency", legend=:bottomright)
h3 = histogram(k21s, label="k21", xlabel="k21 [1/s]", ylabel="Frequency", legend=:bottomright)
h4 = histogram(k13s, label="k13", xlabel="k13 [1/s]", ylabel="Frequency", legend=:bottomright)
h5 = histogram(k31s, label="k31", xlabel="k31 [1/s]", ylabel="Frequency", legend=:bottomright)
h6 = histogram(V1s, label="V1", xlabel="V1 [litres]", ylabel="Frequency", legend=:bottomright)
plot(h1, h2, h3, h4, h5, h6, size=(1200, 800),layout=grid(3, 2))





## Save results in CSV files
using CSV, DataFrames

df_losses = DataFrame("Epoch" => 1:length(losses), "Losses per epoch" => losses, "Logarithmic losses per epoch" => log.(losses))
# CSV.write("csv/losses_nn.csv", df_losses)

df_mdalelosses = DataFrame("Epoch" => 1:length(losses_mdales), "MdALE loss per epoch" => losses_mdales, "Logarithmic MdALE loss per epoch" => log.(losses_mdales))
# CSV.write("csv/mdalelosses_nn.csv", df_mdalelosses)


y_nn = reduce(vcat, y_pred)[2:end]

yin = CSV.read("csv/predicted_observed_conc.csv", DataFrame)
yin = yin[2:end, :]
# yin = Matrix(yin)

y_obs = yin[:, 1]
y_symreg = yin[:, 2]
y_eleveld = yin[:, 3]

df_y = DataFrame("Observed concentration" => y_obs, "Predicted concentration symbolic regression" => y_symreg, "Predicted concentration Eleveld" => y_eleveld, "Predicted concentration neural network" => y_nn)
# CSV.write("csv/y_all.csv", df_y)

# Save PK parameters to file
df_pkparams = DataFrame("k10 [1/s]" => k10s, "k12 [1/s]" => k12s, "k21 [1/s]" => k21s, "k13 [1/s]" => k13s, "k31 [1/s]" => k31s, "V1 [litres]" => V1s)
# CSV.write("csv/pkparams_nn.csv", df_pkparams)



#########################################################################################################################
# using Pkg
# cd(@__DIR__)
# Pkg.activate(".")
# using CSV, DataFrames
# using Flux, Plots, Random, StatsBase

# seed = 12345
# Random.seed!(seed) # for reproducibility
# include("fastpksim.jl")
# include("get_elevelddata.jl")
# include("fcts.jl")

# # Get data
# x, y, _ = get_data(1, 1031) # get eleveld data

# ## Read results from CSV file
# df_losses = CSV.read("csv/losses_nn.csv", DataFrame)
# losses2 = df_losses[:, 2]

# df_y = CSV.read("csv/y_all.csv", DataFrame)
# y_obs2 = df_y[:, 1]
# y_symreg2 = df_y[:, 2]
# y_eleveld2 = df_y[:, 3]
# y_nn2 = df_y[:, 4]

# # Rebuild y_nn and y_obs
# ynn_rebuild = Vector{Vector{Float64}}(undef, 1031)
# yobs_rebuild = Vector{Vector{Float64}}(undef, 1031)
# yeleveld_rebuild = Vector{Vector{Float64}}(undef, 1031)
# ysymreg_rebuild = Vector{Vector{Float64}}(undef, 1031)
# firstind = 1
# lastind = 9
# ynn_rebuild[1] = y_nn2[firstind:lastind]
# yobs_rebuild[1] = y_obs2[firstind:lastind]
# yeleveld_rebuild[1] = y_eleveld2[firstind:lastind]
# ysymreg_rebuild[1] = y_symreg2[firstind:lastind]
# firstind = lastind + 1
# for i = 2:1031
#     lastind = firstind + length(x[i].youts)-1
#     ynn_rebuild[i] = y_nn2[firstind:lastind]
#     yobs_rebuild[i] = y_obs2[firstind:lastind]
#     yeleveld_rebuild[i] = y_eleveld2[firstind:lastind]
#     ysymreg_rebuild[i] = y_symreg2[firstind:lastind]
#     firstind = lastind + 1
# end

# print("Mean(MdALE) over the data set is: ", MeanMdALE(yobs_rebuild, ynn_rebuild))
# print("Mean(MdLE) over the data set is: ", MeanMdLE(yobs_rebuild, ynn_rebuild))
# print("Mean(MdAPE) over the data set is: ", MeanMdAPE(yobs_rebuild, ynn_rebuild))
# print("Mean(MdPE) over the data set is: ", MeanMdPE(yobs_rebuild, ynn_rebuild))

# # Save mdale losses to file
# mdale_nn = MdALE.(yobs_rebuild, ynn_rebuild)
# mdale_eleveld = MdALE.(yobs_rebuild, yeleveld_rebuild)
# mdale_symreg = MdALE.(yobs_rebuild, ysymreg_rebuild)

# df_mdale = DataFrame("MdALE symbolic regression" => mdale_symreg, "MdALE Eleveld" => mdale_eleveld, "MdALE neural network" => mdale_nn)
# # CSV.write("csv/mdale_all.csv", df_mdale)



