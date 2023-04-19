using Statistics
using Plots

const FIGURE_POSTFIXES = [".png", ".jpg", ".jpeg"]

function recover_original_series(scaled_series::Vector{Float64}, original_ymin::Float64, original_ymax::Float64)
    scaled_ymin = minimum(scaled_series)
    scaled_ymax = maximum(scaled_series)
    recovery_factor = (original_ymax - original_ymin) / (scaled_ymax - scaled_ymin)
    original_series = original_ymin .+ recovery_factor .* (scaled_series .- scaled_ymin)
end

function plot_series(x_series::Vector{Float64}, y_series::Vector{Float64}, save_name::String="tmp_series.png")
    p = plot(x_series, y_series, xlabel="X", ylabel="PnL", size=(1800, 500), label="Series", legend=:bottomright)
    scatter!(x_series, y_series, label=nothing, markersize=2, markercolor=:red, legend=:bottomright)
    if any([endswith(save_name, postfix) for postfix in FIGURE_POSTFIXES])
        savefig(p, save_name)
    else
        savefig(p, string(save_name, ".png"))
    end
    p
end

function plot_series_step(x_series::Vector{Float64}, y_series::Vector{Float64}, scatter_count::Int64=64, save_name::String="tmp_series.png")
    p = plot(y_series, size=(1800, 500), xlabel="Date", ylabel="Series", label="Series", legend=:bottomright)

    # scatter!(y_series, label="Points", markersize=2, markercolor=:red, legend=:bottomright)
    if length(x_series) != length(y_series)
        throw(DomainError("Length of x_series and y_series must be the same."))
    end
    step = max(length(y_series) ÷ scatter_count, 1)
    # xticks!((1:step:length(y_series), x_series[1:step:end]), rotation=45, ha="right", va="top", fontsize=20)
    for i in 1:step:length(y_series)
        scatter!([i], [y_series[i]], markercolor="red", markersize=2, label=nothing)
    end
    if any([endswith(save_name, postfix) for postfix in FIGURE_POSTFIXES])
        savefig(p, save_name)
    else
        savefig(p, string(save_name, ".png"))
    end
    p
end