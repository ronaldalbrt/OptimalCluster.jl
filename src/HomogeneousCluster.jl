module HomogeneousCluster

    using JuMP, HiGHS
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
    function langrangian(u::AbstractVector, data::ClusterProblemData)::Vector
        n_points = data.n_points
        n_cluster_candidates = data.n_cluster_candidates
        d = data.d
        m = data.n_clusters
        
        penalized_distances = d .- u
        
        sum_penalized_distances = sum(min.(penalized_distances, 0), axis = 1)
        m_lowest_distances = partialsortperm(sum_penalized_distances, 1:m)
        
        x = zeros(n_points, n_cluster_candidates)
        for i in m_lowest_distances
            x[i, i] = 1
            
            for j in 1:n_cluster_candidates
                if penalized_distances[i, j] < 0
                    x[i, j] = 1
                end
            end
        end

        return x, (sum(u) + sum((d .- u).*x))
    end

    # --------------------------------------------------------------
    # Subgradient Optimization of the Lagrange multipliers
    # --------------------------------------------------------------
    # Parameters:
    # inital_u: Initial Vector of Lagrange multipliers
    # d: Vector of distances between all data points
    # m: Number of clusters
    # θ: Smoothing parameter for the step
    # n_steps: Number of steps to perform
    # upper_bound: Upper bound for the Lagrangian Problem
    # ε: Hyperparameter for the step size calculation
    # --------------------------------------------------------------
    # Return: The solution found for the Lagragian Dual of the Homogeneous Clustering Problem
    # --------------------------------------------------------------
    function subgradient_algorithm(initial_u::AbstractVector, data::ClusterProblemData, θ::Real, n_steps::Integer, upper_bound::Integer, ε::Real)::Vector
        u = initial_u

        prev_subgradient = zeros(size(u))
        for i in 1:n_steps
            x, Z_u = langrangian(u, data)
            constraint_values = (1 .- sum(x, dims = 2))
            subgradient = constraint_values + θ .* prev_subgradient

            step_size = ε*(upper_bound - Z_u)/(constraint_values'*constraint_values) 

            u = u + step_size .* subgradient

            prev_subgradient = subgradient
        end

        return u
    end

    # --------------------------------------------------------------
    # Definition of the Jump formulation for the Homogeneous Clustering Problem
    # --------------------------------------------------------------
    # Parameters:
    # data: A struct containing the data for the Homogeneous Clustering Problem
    # --------------------------------------------------------------
    # Return: The Jump formulation for the Homogeneous Clustering Problem
    # --------------------------------------------------------------
    function jump_formulation(data::ClusterProblemData)
        n = data.n_points
        m = data.n_clusters
        n_cluster_candidates = data.n_cluster_candidates
        d = data.d

        model = Model(HiGHS.Optimizer)
        @variable(model, x[1:n, 1:n_cluster_candidates], Bin)

        @constraint(model, sum(x, dims = 2) .== 1)
        @constraint(model, sum([x[i,i] for i in 1:n_cluster_candidates]) == m)

        @constraint(model, [j = 1:n_cluster_candidates], x[:,j] .<= x[j,j])

        @objective(model, Min, sum(d.*x))

        return model
    end
end