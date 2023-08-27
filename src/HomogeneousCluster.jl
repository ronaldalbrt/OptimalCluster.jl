module HomogeneousCluster
    using JuMP, Gurobi
    # --------------------------------------------------------------
    # Definition of the Data for the Homogeneous Clustering Problem
    # --------------------------------------------------------------
    # Parameters:
    # n_points: Number of data points
    # n_cluster_candidates: Number of cluster candidates
    # n_clusters: Number of clusters
    # d: Vector of distances between all data points
    # --------------------------------------------------------------
    # Return: A struct containing the data for the Homogeneous Clustering Problem
    # --------------------------------------------------------------
    struct ClusterProblemData
        n_points::Integer
        n_cluster_candidates::Integer
        n_clusters::Integer
        d::AbstractMatrix
    end

    # --------------------------------------------------------------
    # Definition of the Lagrangian Relaxation for the Homogeneous Clustering Problem
    # --------------------------------------------------------------
    # Parameters:
    # u: Vector of Lagrange multipliers
    # d: Vector of distances between all data points
    # m: Number of clusters
    # --------------------------------------------------------------
    # Return: The solution for the Lagragian Relaxed Homogeneous Clustering Problem
    # --------------------------------------------------------------
    function lagrangian(u::AbstractVector, data::ClusterProblemData)::Tuple{Matrix, Float64} 
        n_points = data.n_points
        n_cluster_candidates = data.n_cluster_candidates
        d = data.d
        m = data.n_clusters
        
        penalized_distances = d .- u
        
        sum_penalized_distances = sum(min.(penalized_distances, 0), dims=1)
        m_lowest_distances = partialsortperm(vec(sum_penalized_distances), 1:m)
        
        x = zeros(n_points, n_cluster_candidates)
        for j in m_lowest_distances
            x[j, j] = 1
            
            for i in 1:n_points
                if penalized_distances[i, j] < 0
                    x[i, j] = 1
                end
            end
        end

        return x, sum(u) + sum((d .- u).*x)
    end

    # --------------------------------------------------------------
    # Subgradient Optimization of the Lagrange multipliers
    # using step_size: μ_k = ε * (lower_bound - Zu) / ||∇u_k||^2
    # --------------------------------------------------------------
    # Parameters:
    # inital_u: Initial Vector of Lagrange multipliers
    # d: Vector of distances between all data points
    # m: Number of clusters
    # θ: Smoothing parameter for the step
    # n_steps: Number of steps to perform
    # lower_bound: Lower bound for the Lagrangian Problem
    # ε: Hyperparameter for the step size calculation
    # --------------------------------------------------------------
    # Return: The solution found for the Lagragian Dual of the Homogeneous Clustering Problem
    # --------------------------------------------------------------
    function subgradient_algorithm(initial_u::AbstractVector, data::ClusterProblemData, θ::Real, n_steps::Integer, lower_bound::Integer, ϵ::Real)
        u = initial_u

        prev_subgradient = zeros(size(u))
        results = []
        for _ in 1:n_steps
            x, Zu = lagrangian(u, data)
            
            push!(results, Zu)
            constraint_values = vec(1 .- sum(x, dims = 2))
            subgradient = constraint_values + θ .* prev_subgradient
            
            step_size = ϵ * (lower_bound - Zu) / sum(subgradient.^2)

            u = u + step_size .* subgradient

            prev_subgradient = subgradient
        end

        return u, results
    end

    # --------------------------------------------------------------
    # Subgradient Optimization of the Lagrange multipliers
    # using step_size: μ_k = μ_0 * ρ^k
    # -------------------------------------------------------------- 
    # Parameters:
    # inital_u: Initial Vector of Lagrange multipliers
    # d: Vector of distances between all data points
    # m: Number of clusters
    # θ: Smoothing parameter for the step
    # n_steps: Number of steps to perform
    # μ: μ parameter for the step size calculation
    # ρ: ρ parameter for the step size calculation
    # --------------------------------------------------------------
    # Return: The solution found for the Lagragian Dual of the Homogeneous Clustering Problem
    # --------------------------------------------------------------lower_bound
    function subgradient_algorithm(initial_u::AbstractVector, data::ClusterProblemData, θ::Real, n_steps::Integer, μ::Real, ρ::Real)
        u = initial_u

        prev_subgradient = zeros(size(u))
        results = []
        for k in 1:n_steps
            x, Zu = lagrangian(u, data)
            
            push!(results, Zu)
            constraint_values = vec(1 .- sum(x, dims = 2))
            subgradient = constraint_values + θ .* prev_subgradient
            
            step_size = μ*ρ^k

            u = u + step_size .* subgradient

            prev_subgradient = subgradient
        end

        return u, results
    end

    # --------------------------------------------------------------
    # Definition of the Jump formulation for the Homogeneous Clustering Problem
    # --------------------------------------------------------------
    # Parameters:
    # data: A struct containing the data for the Homogeneous Clustering Problem
    # --------------------------------------------------------------
    # Return: The Jump formulation for the Homogeneous Clustering Problem
    # --------------------------------------------------------------
    function jump_formulation(data::ClusterProblemData, u)
        n = data.n_points
        m = data.n_clusters
        n_cluster_candidates = data.n_cluster_candidates
        d = data.d

        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "OutputFlag", 0)
        set_optimizer_attribute(model, "TimeLimit", 100)
        set_optimizer_attribute(model, "MIPGap", 0.001)
        set_optimizer_attribute(model, "Threads", min(length(Sys.cpu_info()),16))

        @variable(model, x[1:n, 1:n_cluster_candidates], Bin)

        @constraint(model, sum(x, dims = 2) .== 1)
        @constraint(model, sum([x[i,i] for i in 1:n_cluster_candidates]) == m)

        @constraint(model, [j = 1:n_cluster_candidates], x[:,j] .<= x[j,j])

        @objective(model, Min, sum(d.*x))
        return model
    end
end