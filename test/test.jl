using Pkg
Pkg.activate(".")
Pkg.instantiate()

using OptimalCluster
using CSV, DataFrames, LinearAlgebra, JuMP, Plots

#Load File
file = ARGS[1]
m = parse(Int64, ARGS[2])

data_df = CSV.read("datasets/$file", DataFrame)

n = size(data_df, 1)
X = Matrix(data_df)
distances = [norm(X[i, :] - X[j, :]) for i in 1:n, j in 1:n]

output = Dict()
# Data
data = OptimalCluster.HomogeneousCluster.ClusterProblemData(n, m, distances)


model = OptimalCluster.HomogeneousCluster.jump_formulation(data)
opt_time = optimize!(model)

output["obj_value"] = objective_value(model)

n_steps = 100
lower_bound = Int64(ceil(output["obj_value"]))
ϵ = .5

gap = 0.1

initial_u = rand(data.n_points)
best_solution, results =  OptimalCluster.HomogeneousCluster.subgradient_algorithm(initial_u, data, .6, n_steps, lower_bound, ϵ; stop_gap=gap)

output["best_solution"] = best_solution
output["results"] = results

using JSON
open("results/$file-$m.json", "w") do f
    JSON.print(f, output)
end