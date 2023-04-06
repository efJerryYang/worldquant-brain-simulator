using Statistics
using Plots

function recover_original_series(scaled_series::Vector{Float64}, original_ymin::Float64, original_ymax::Float64)
    scaled_ymin = minimum(scaled_series)
    scaled_ymax = maximum(scaled_series)
    recovery_factor = (original_ymax - original_ymin) / (scaled_ymax - scaled_ymin)
    return original_series = original_ymin .+ recovery_factor .* (scaled_series .- scaled_ymin)
end

function plot_series(x_series::Vector{Float64}, y_series::Vector{Float64})
    p = plot(x_series, y_series, xlabel="X", ylabel="PnL", size=(1800, 500), label="Series", legend=:bottomright)
    scatter!(x_series, y_series, label="Points", markersize=2, markercolor=:red, legend=:bottomright)
    savefig(p, "tmp_series.png")
    return p
end