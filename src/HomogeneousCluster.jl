module HomogeneousCluster

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
    function langrangian(u::AbstractVector, d::AbstractMatrix, m::Integer)::Vector
        n_points = size(d, 1)
        n_cluster_candidates = size(d, 2)
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

        return x
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
    # --------------------------------------------------------------
    # Return: The solution found for the Lagragian Dual of the Homogeneous Clustering Problem
    # --------------------------------------------------------------
    function subgradient_algorithm(initial_u::AbstractVector, d::AbstractMatrix, m::Integer, θ::Real, n_steps::Integer)::Vector
        u = initial_u

        prev_subgradient = zeros(size(u))
        for i in 1:n_steps
            x = langrangian(u, d, m)
            subgradient = (1 .- sum(x, dims = 2)) + θ .* prev_subgradient

            step_size = 1 / i

            u = u + step_size .* subgradient

            prev_subgradient = subgradient
        end

        return u
    end
end