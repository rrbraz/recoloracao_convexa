module BranchAndBound
    using JuMP
    using Gurobi
    using MathOptInterface
    using DataStructures
    using MathProgBase
    using CPUTime

    mutable struct Input
        n_vertices::Int32
        n_cores::Int32
        c::Array{Int}
    end

    mutable struct BB_Node
        varToFix::CartesianIndex
        fix1::Bool
        parent::Union{BB_Node, Nothing}
        left_child::Union{BB_Node, Nothing}
        right_child::Union{BB_Node, Nothing}
        LB::Float64

        function BB_Node(varToFix, fix1, parent, LB)
            return new(varToFix, fix1, parent, nothing, nothing, LB)
        end
    end

    mutable struct Config
        turn_heuristic_on::Bool
        turn_cuts_on::Bool
        num_max_cuts_per_iter::Int16
        violation_threshold::Float64
        time_limit::Float64

        function Config()
            return new(true, true, 10, 0.001, 1800.0)
        end
    end

    mutable struct Statistics
        n_BB_nodes::Int32
        heuristic_sol_val::Float64
        objective_value::Float64
        first_linear_relax_val::Float64
        spent_time_in_seconds::Float64

        function Statistics()
            return new(0, 0, 0.0, 0.0, 0.0)
        end
    end

    ϵ = 0.000001
    input = nothing
    model = nothing
    x = nothing
    config = Config()
    statistics = Statistics()

    function init()
        global input = nothing
        global model = nothing
        global x = nothing
        global config = Config()
        global statistics = Statistics()
    end

    # instâncias obtidas em http://artemisa.unicauca.edu.co/~johnyortega/instances_01_KP/
    function read_input(nome_arq)
        linha_arq = []

        open(nome_arq) do arq
            linha_arq = readlines(arq)
        end

        n_vertices = parse(Int, split(linha_arq[1])[1])
        n_cores = parse(Int, split(linha_arq[1])[2])

        c = Array{Int}(undef, n_vertices)

        for i in 1:n_vertices
            linha = parse(Int, linha_arq[1+i])
            c[i] = linha
        end

        global input = Input(n_vertices, n_cores, c)
    end

    function build_initial_model(lp_relax)
        # criando um model "vazio" no gurobi
        global model = direct_model(Gurobi.Optimizer())

        # setando alguns parâmetros específicos do Gurobi
        set_optimizer_attribute(model, "LogToConsole", 0)
        set_optimizer_attribute(model, "Threads", 1)
        set_optimizer_attribute(model, "Method", 1)

        # --- adicionando as variáveis no modelo ---
        global x = Array{Any}(undef, input.n_vertices, input.n_cores)
        for v in 1:input.n_vertices
            for k in 1:input.n_cores
                if lp_relax
                    x[v,k] = @variable(model, upper_bound=1, lower_bound=0)
                else
                    x[v,k] = @variable(model, binary=true)
                end
            end
        end
        # ------------------------------------------

        # --- adicionando a função objetivo ao model ---
        trocas_de_cor = AffExpr()
        for v in 1:input.n_vertices
            for k in 1:input.n_cores
                if k != input.c[v]
                    add_to_expression!(trocas_de_cor, x[v,k])
                end
            end
        end

        @objective(model, Min, trocas_de_cor)
        # ----------------------------------------------

        # --- adicionando a restrição de uma unica cor por vertice ---
        for v in 1:input.n_vertices
            soma_cores_vertice = AffExpr()
            for k in 1:input.n_cores
                add_to_expression!(soma_cores_vertice, x[v,k])
            end
            @constraint(model, soma_cores_vertice == 1)
        end
        # ------------------------

        # --- adicionando as restrições de convexidade ---
        for k in 1:input.n_cores
            for p in 1:input.n_vertices-2
                for q in p+1:input.n_vertices-1
                    for r in q+1:input.n_vertices
                        @constraint(model, x[p,k] - x[q,k] + x[r,k] <= 1)
                    end
                end
            end
        end
        # ------------------------
    end

    function solve_LP(no)
        optimize!(model)
        return termination_status(model)
    end

    function add_BB_node(L, no)
        push!(L, no)
    end

    function select_BB_node(L)
        i = 1
        i_selected_node = -1
        selected_node = nothing
        min_LB = Inf

        for node in L
            if node.LB < min_LB
                min_LB = node.LB
                selected_node = node
                i_selected_node = i
            end
            i += 1
        end

        deleteat!(L, i_selected_node)
        return selected_node
    end

    function select_frac_var()
        i_mais_frac = CartesianIndex(-1, -1)
        mais_frac = 0

        for v in 1:input.n_vertices
            for k in 1:input.n_cores
                valor = value(x[v,k])

                if abs(valor - 0.5) < abs(mais_frac - 0.5)
                    i_mais_frac = CartesianIndex(v, k)
                    mais_frac = valor
                end
            end
        end

        if mais_frac < ϵ || mais_frac > 1-ϵ
            return -1
        else
            return i_mais_frac
        end
    end

    function is_integer()
        return select_frac_var() == -1
    end

    function copy_solution(x_int)
        for v in 1:input.n_vertices
            for k in 1:input.n_cores
                if value(x[v,k]) > ϵ
                    x_int[v, k] = 1
                else
                    x_int[v, k] = 0
                end
            end
        end
    end

    function add_branch_constraints(no)
        if isnothing(no.parent) == false
            if no.fix1
                for k in 1:input.n_cores
                    # fixa todas outras cores do nó em 0
                    if k != no.varToFix[2]
                        v = no.varToFix[1]
                        set_upper_bound(x[v, k], 0)
                    end
                end
                set_lower_bound(x[no.varToFix], 1)
            else
                set_upper_bound(x[no.varToFix], 0)
            end

            add_branch_constraints(no.parent)
        end
    end

    function clear_branch_constraints()
        for v in 1:input.n_vertices
            for k in 1:input.n_cores
                set_upper_bound(x[v,k], 1)
                set_lower_bound(x[v,k], 0)
            end
        end
    end

    function create_BB_node(varToFix, fix1, parent, LB)
        global statistics.n_BB_nodes += 1
        no = BB_Node(varToFix, fix1, parent, LB)
        return no
    end

    function solve_IP()
        build_initial_model(false)
        CPUtic()
        optimize!(model)
        statistics.spent_time_in_seconds += CPUtoq()

        # recupera e imprime a solução
        sol = zeros(input.n_vertices,input.n_cores)
        copy_solution(sol)
        statistics.objective_value = solution_value(sol)
        statistics.n_BB_nodes = MOI.get(model, Gurobi.ModelAttribute("NodeCount"))

        print_solution(sol)
    end

    function solution_value(sol)
        val = 0
        for v in 1:input.n_vertices
            for k in 1:input.n_cores
                if k != input.c[v]
                    val += sol[v,k]
                end
            end
        end
        return val
    end

    function heuristic()
        # constroi configuracao inicial trivial com todos vertices da mesma cor
        sol = zeros(input.n_vertices, input.n_cores)
        for v in 1:input.n_vertices
            sol[v, input.c[1]] = 1
        end
        value = solution_value(sol)

        if config.turn_heuristic_on == false
            return sol, value
        end
        
        # uma heuristica levemente melhor que mantém as cores quando aparecem pela primeira vez
        cores_utilizadas = Set()
        ultima_cor = 0
        sol = zeros(input.n_vertices, input.n_cores)
        for v in 1:input.n_vertices
            if input.c[v] ∉ cores_utilizadas
                ultima_cor = input.c[v]
                push!(cores_utilizadas, ultima_cor)
            end
            sol[v, ultima_cor] = 1
        end
        value = solution_value(sol)

        statistics.heuristic_sol_val = value
        return sol, value
    end

    function branch_and_bound()
        # 0. Initialize
        CPUtic()
        build_initial_model(true)
        n_added_cuts_curr_iter = 0
        best_sol, z_ = heuristic()
        n_0 = create_BB_node(CartesianIndex(-1, -1), false, nothing, 0)
        L = []
        add_BB_node(L, n_0)
        is_root_node = true
        root_node = nothing

        # ----- 1. Terminate ? -----
        while isempty(L) == false
            
            # --- 1.1 Global LB >= UB ? ---
            if is_root_node == false
                if root_node.LB >= z_ + ϵ
                    break
                end
            end

            # --- 1.2 is the time limit exceeded? ---
            global statistics.spent_time_in_seconds += CPUtoq()
            CPUtic()
            if statistics.spent_time_in_seconds > config.time_limit
                time_limit_exceeded(best_sol, z_, root_node.LB)
                return
            end

            # ----- 2. Select node -----
            n_i = select_BB_node(L)
            clear_branch_constraints()
            add_branch_constraints(n_i)

            # ----- 3. Bound -----
            status = solve_LP(n_i)
            # println("Partial solution:")
            # show(IOContext(stdout, :limit => false), "text/plain", map(value, x))
            # println()

            if status == MathOptInterface.INFEASIBLE
                continue
            else
                z_i = objective_value(model)
                n_i.LB = z_i
                update_parent_LB(n_i)

                if is_root_node
                    statistics.first_linear_relax_val = z_i
                    root_node = n_i
                    is_root_node = false
                end
            end

            # ----- 4. Prune -----
            if z_i >= z_ + ϵ
                continue
            end

            if is_integer()
                z_ = z_i
                copy_solution(best_sol)
                continue
            end

            # ----- 6. Branch -----
            i_var = select_frac_var()
            n_i.left_child = create_BB_node(i_var, true, n_i, z_i) # fix 1
            n_i.right_child = create_BB_node(i_var, false, n_i, z_i) # fix 0
            add_BB_node(L, n_i.left_child)
            add_BB_node(L, n_i.right_child)
        end

        statistics.objective_value = z_
        print_solution(best_sol)
    end

    function update_parent_LB(node)        
        parent = node.parent
        if !isnothing(parent)
            parent.LB = min(parent.left_child.LB, parent.right_child.LB)
            update_parent_LB(parent)
        end
    end

    function print_solution(sol)
        println()
        objval = solution_value(sol)

        println()
        println_in_yellow(string("Number of vertice: ", input.n_vertices))
        println_in_yellow(string("Number of colors: ", input.n_cores))
        println_in_yellow(string("Spent time in seconds: ", statistics.spent_time_in_seconds))
        println_in_yellow(string("Objective value: ", objval))
        println(string("Heuristic value: ", statistics.heuristic_sol_val))
        println(string("First linear relaxation value: ", statistics.first_linear_relax_val))
        println(string("Number of BB nodes: ", statistics.n_BB_nodes))
        println(statistics)

        println()
        println(string("Coloração original:  ", input.c))

        total_trocas=0
        c2 = Array{Int}(undef, input.n_vertices)
        for v in 1:input.n_vertices
            for k in 1:input.n_cores
                if value(sol[v,k]) >= 1
                    c2[v] = k
                    if k != input.c[v]
                        total_trocas += 1
                    end
                end
            end
        end

        println(string("Coloração escolhida: ", c2))
    end

    function time_limit_exceeded(best_sol, z_, LB)
        println()
        println_in_yellow("*** TIME LIMIT EXCEEDED ***")
        println_in_yellow(string("LB: ", LB))
        println_in_yellow(string("UB: ", z_))
        print_solution(best_sol)
    end

    function print_in_yellow(texto)
        print("\e[1m\e[38;2;255;225;0;249m", texto)
    end

    function println_in_yellow(texto)
        println("\e[1m\e[38;2;255;255;0;249m", texto)
    end

    function exec_test(instancia, turn_heuristic_on)
        init()

        arq_instancia = string("instancias/", instancia)
        read_input(arq_instancia)

        config.turn_heuristic_on = turn_heuristic_on
        branch_and_bound()
        # solve_IP()
    end

    function exec_tests()
        results = []
        for instance_size in [10, 20, 30, 40, 50]
            for n_colors in 2:10
                instance = "rand_$(instance_size)_$(n_colors).txt"
                exec_test(instance, true)
                a1 = instance_size
                b1 = n_colors
                c1 = statistics.objective_value
                d1 = statistics.heuristic_sol_val
                e1 = statistics.first_linear_relax_val
                f1 = statistics.spent_time_in_seconds
                g1 = statistics.n_BB_nodes

                push!(results, string(a1, ";", b1, ";", c1, ";", d1, ";", e1, ";", f1, ";", g1))
            end
        end

        for r in results
            println(r)
        end
    end

    exec_tests()
end;
