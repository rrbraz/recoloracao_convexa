module BranchAndBound
    using JuMP
    using Gurobi
    using MathOptInterface
    using DataStructures
    using MathProgBase
    using CPUTime

    @enum Mode basic user_cuts lazy_constraints

    mutable struct Input
        n_vertices::Int32
        n_cores::Int32
        c::Array{Int}
    end


    mutable struct Config
        turn_heuristic_on::Bool
        time_limit::Float64
        mode::Mode

        function Config()
            return new(true, 1800.0, basic)
        end
    end

    mutable struct Statistics
        constrs::Int64
        vars::Int64
        n_BB_nodes::Int32
        itercount::Int64
        n_cuts::Int32
        nonzeros::Int64
        objective_value::Float64
        spent_time_in_seconds::Float64
        runtime::Float64

        function Statistics()
            return new(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0)
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
        set_optimizer_attribute(model, "LogToConsole", 1)
        set_optimizer_attribute(model, "Threads", 1)
        set_optimizer_attribute(model, "Method", 1)
        set_optimizer_attribute(model, "TimeLimit", config.time_limit)

        # --- adicionando as variáveis no modelo ---
        global x = Array{Any}(undef, input.n_vertices, input.n_cores)
        for v in 1:input.n_vertices
            for k in 1:input.n_cores
                if lp_relax
                    x[v,k] = @variable(model, upper_bound=1, lower_bound=0, base_name="x_$(v)_$(k)")
                else
                    x[v,k] = @variable(model, binary=true, base_name="x_$(v)_$(k)")
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
        
        if config.mode != lazy_constraints
            # --- adicionando as restrições de convexidade apenas se não for Lazy Constraints ---
            for k in 1:input.n_cores
                for p in 1:input.n_vertices-2
                    for q in p+1:input.n_vertices-1
                        for r in q+1:input.n_vertices
                            @constraint(model, x[p,k] - x[q,k] + x[r,k] <= 1)
                        end
                    end
                end
            end
        end
        # ------------------------
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

    function build_sep_ineq_convex(v, k)
        mais = fill(-Inf, length(v))
        menos = fill(-Inf, length(v))
        ant_mais = zeros(Int, length(v))
        ant_menos = zeros(Int, length(v))

        mais[1] = v[1]
        p = 1
        q = 1

        for r in 2:length(v)
            if v[r] > menos[q] + v[r]
                ant_mais[r] = 0
            else
                ant_mais[r] = q
            end

            mais[r] = max(v[r], menos[q] + v[r])
            

            menos[r] = mais[p] - v[r]
            ant_menos[r] = p

            if mais[r] > mais[p]
                p = r
            end
            if menos[r] > menos[q]
                q = r
            end
        end
        
        if (mais[p] > 1 + ϵ)
            sinal = 1
            expr = AffExpr()
            while p != 0
                add_to_expression!(expr, sinal, x[p,k])
                if sinal == 1
                    sinal = -1
                    p = ant_mais[p]
                else
                    sinal = 1
                    p = ant_menos[p]
                end
            end
            constraint = @build_constraint(expr <= 1)
            return constraint
        end
    end

    function callback(cb_data, cb_where::Cint)
        if cb_where != GRB_CB_MIPSOL && cb_where != GRB_CB_MIPNODE
            return
        end
        # You can query a callback attribute using GRBcbget
        if cb_where == GRB_CB_MIPNODE
            resultP = Ref{Cint}()
            GRBcbget(cb_data, cb_where, GRB_CB_MIPNODE_STATUS, resultP)
            if resultP[] != Gurobi.GRB_OPTIMAL
                return  # Solution is something other than optimal.
            end
        end

        # Before querying `callback_value`, you must call:
        Gurobi.load_callback_variable_primal(cb_data, cb_where)

        x_val = Array{Any}(undef, input.n_vertices, input.n_cores)
        for k in 1:input.n_cores
            for v in 1:input.n_vertices
                x_val[v, k] = callback_value(cb_data, x[v, k])        
            end

            con = build_sep_ineq_convex(x_val[:,k], k) 
            if con !== nothing
                statistics.n_cuts += 1
                if config.mode == lazy_constraints
                    MOI.submit(model, MOI.LazyConstraint(cb_data), con)
                elseif config.mode == user_cuts
                    MOI.submit(model, MOI.UserCut(cb_data), con)
                end
            end
        end
    end

    function solve_IP()
        build_initial_model(false)

        if config.mode != basic
            if config.mode == lazy_constraints
                MOI.set(model, MOI.RawParameter("LazyConstraints"), 1)
            end
            MOI.set(model, Gurobi.CallbackFunction(), callback)
        end

        CPUtic()
        optimize!(model)
        statistics.spent_time_in_seconds += CPUtoq()

        # recupera e imprime a solução
        sol = zeros(input.n_vertices,input.n_cores)
        copy_solution(sol)
        statistics.objective_value = solution_value(sol)
        statistics.n_BB_nodes = MOI.get(model, Gurobi.ModelAttribute("NodeCount"))
        statistics.itercount = MOI.get(model, Gurobi.ModelAttribute("IterCount"))
        statistics.constrs = MOI.get(model, Gurobi.ModelAttribute("NumConstrs"))
        statistics.vars = MOI.get(model, Gurobi.ModelAttribute("NumVars"))
        statistics.nonzeros = MOI.get(model, Gurobi.ModelAttribute("NumNZs"))
        statistics.runtime = MOI.get(model, Gurobi.ModelAttribute("Runtime"))

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

    function print_solution(sol)
        println()
        objval = solution_value(sol)

        println()
        println_in_yellow(string("Number of vertice: ", input.n_vertices))
        println_in_yellow(string("Number of colors: ", input.n_cores))
        println_in_yellow(string("Spent time in seconds: ", statistics.spent_time_in_seconds))
        println_in_yellow(string("Objective value: ", objval))
        println(string("Number of BB nodes: ", statistics.n_BB_nodes))
        println(string("Number of cuts added: ", statistics.n_cuts))
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

    function print_in_yellow(texto)
        print("\e[1m\e[38;2;255;225;0;249m", texto)
    end

    function println_in_yellow(texto)
        println("\e[1m\e[38;2;255;255;0;249m", texto)
    end

    function exec_test(instancia; turn_heuristic_on=true, mode=basic)
        init()

        arq_instancia = string("instancias/", instancia)
        read_input(arq_instancia)

        config.turn_heuristic_on = turn_heuristic_on
        config.mode = mode
        solve_IP()
    end

    function exec_tests(;turn_heuristic_on=true, mode=basic)
        open("$mode.csv", "w") do io
            write(io, "size;colors;obj_val;vars;constrs;nonzeros;bb_nodes;itercount;cuts;runtime\n")
        
            for instance_size in [10, 20, 30, 40, 50]
                for n_colors in 2:10
                    instance = "rand_$(instance_size)_$(n_colors).txt"
                    exec_test(instance, turn_heuristic_on=turn_heuristic_on, mode=mode)
                    line = string(
                        instance_size, ";", 
                        n_colors, ";", 
                        statistics.objective_value, ";", 
                        statistics.vars, ";", 
                        statistics.constrs, ";", 
                        statistics.nonzeros, ";", 
                        statistics.n_BB_nodes, ";", 
                        statistics.itercount, ";", 
                        statistics.n_cuts, ";", 
                        statistics.spent_time_in_seconds, "\n", 
                    )
                    write(io, line)
                end
            end

        end
    end

    exec_tests(mode=basic)
end;
