module HomogeneousCluster
    using JuMP, Gurobi
    # --------------------------------------------------------------
    # Definition of the Data for the Homogeneous Clustering Problem
    # --------------------------------------------------------------
    # Parameters:
    # n_points: Number of data points
    # n_clusters: Number of clusters
    # d: Matrix of distances between all data points
    # --------------------------------------------------------------
    # Return: A struct containing the data for the Homogeneous Clustering Problem
    # --------------------------------------------------------------
    struct ClusterProblemData
        n_points::Integer
        n_clusters::Integer
        d::AbstractMatrix
    end

    # --------------------------------------------------------------
    # Auxiliary function to build a solution for a set of clusters
    # --------------------------------------------------------------
    # Parameters:
    # clusters: Vector of clusters
    # data: A struct containing the data for the Homogeneous Clustering Problem
    # --------------------------------------------------------------
    # Return: Solution for the Homogeneous Clustering Problem with the predefined clusters
    # --------------------------------------------------------------
    function _build_solution_from_clusters(clusters::AbstractVector, data::ClusterProblemData)
        distance_matrix = data.d
        n_points = data.n_points

        solution = zeros(n_points, n_points)
        for i in 1:n_points
            if i ∉ clusters
                argmin = Base.argmin([distance_matrix[i, j] for j in clusters])
                solution[i, clusters[argmin]] = 1
            else
                solution[i, i] = 1
            end
        end

        return solution
    end

    # --------------------------------------------------------------
    # Primal Search Heuristic for the Homogeneous Clustering Problem
    # --------------------------------------------------------------
    # Parameters:
    # x: Matrix of cluster assignments
    # --------------------------------------------------------------
    # Return: Primal solution for the Homogeneous Clustering Problem
    # --------------------------------------------------------------
    function primal_search(x::Matrix, u::AbstractVector, data::ClusterProblemData)
        distance_matrix = data.d
        n_points = data.n_points
        m = data.n_clusters

        diag_x = [x[i,i] for i in 1:n_points]
        clusters = findall(y -> y == 1, diag_x)

        curr_solution = _build_solution_from_clusters(clusters, data)
        
        opt_value = sum(distance_matrix.*curr_solution)

        intercluster_distances = [i != j ? distance_matrix[i, j] : Inf64 for i in clusters, j in clusters]
        lowest_intercluster_distance = argmin(intercluster_distances)[1]

        penalized_distances = distance_matrix .- u
        sum_penalized_distances = sum(min.(penalized_distances, 0), dims=1)
        lowest_distances = partialsortperm(vec(sum_penalized_distances), 1:2*m)

        for i in lowest_distances
            if i ∉ clusters
                temp_clusters = copy(clusters)
                temp_clusters[lowest_intercluster_distance] = i

                intermediate_solution = _build_solution_from_clusters(temp_clusters, data)
                intermediate_opt_value = sum(distance_matrix.*intermediate_solution)

                if intermediate_opt_value < opt_value
                    clusters = temp_clusters
                    curr_solution = intermediate_solution
                    opt_value = intermediate_opt_value
                end
            end
        end

        return curr_solution, opt_value
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
        d = data.d
        m = data.n_clusters
        
        penalized_distances = d .- u
        
        sum_penalized_distances = sum(min.(penalized_distances, 0), dims=1)
        m_lowest_distances = partialsortperm(vec(sum_penalized_distances), 1:m)
        
        x = zeros(n_points, n_points)
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
    function subgradient_algorithm(initial_u::AbstractVector, data::ClusterProblemData, θ::Real, n_steps::Integer, lower_bound::Integer, ϵ::Real; stop_gap::Real = 0)
        u = initial_u

        prev_subgradient = zeros(size(u))
        results = []
        best_x =  Matrix{Float64}(undef, 0, 0)
        best_solution = Inf64
        best_dual = -Inf64
        for _ in 1:n_steps
            x, Zu = lagrangian(u, data)
            
            constraint_values = vec(1 .- sum(x, dims = 2))
            subgradient = constraint_values + θ .* prev_subgradient
            
            step_size = ϵ * (lower_bound - Zu) / sum(subgradient.^2)

            x_feasible, Z = primal_search(x, u, data)

            u = u + step_size .* subgradient

            if Z < best_solution
                best_x = x_feasible
                best_solution = Z
            end

            if Zu > best_dual
                best_dual = Zu
            end

            if abs(best_solution - best_dual)/best_solution <= stop_gap
                break
            end

            prev_subgradient = subgradient
            
            push!(results, (Z, Zu))
        end

        return best_x, results
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
        d = data.d

        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "OutputFlag", 0)
        set_optimizer_attribute(model, "TimeLimit", 100)
        set_optimizer_attribute(model, "MIPGap", 0.0001)
        set_optimizer_attribute(model, "Threads", min(length(Sys.cpu_info()), 16))

        @variable(model, x[1:n, 1:n], Bin)

        @constraint(model, sum(x, dims = 2) .== 1)
        @constraint(model, sum([x[i,i] for i in 1:n]) == m)

        @constraint(model, [j = 1:n], x[:,j] .<= x[j,j])

        @objective(model, Min, sum(d.*x))
        return model
    end
end