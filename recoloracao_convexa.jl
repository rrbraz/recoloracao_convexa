using JuMP
using Gurobi
using MathOptInterface

# Formato dos arquivos de entrada:
# n_vertices n_cores
# c_1 (cores dos vértices)
# c_2
# ...
# c_n
function le_dados_entrada(nome_arq)
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

    return n_vertices, n_cores, c
end

function resolve_recoloracao_convexa(n_vertices, n_cores, c)
    # criando um modelo "vazio" no gurobi
    modelo = Model(Gurobi.Optimizer)

    # --- adicionando as variáveis no modelo ---
    x = Dict()
    for v in 1:n_vertices
        for k in 1:n_cores
            x[v,k] = @variable(modelo, binary=true)
        end
    end

    # --- adicionando a função objetivo ao modelo ---
    trocas_de_cor = AffExpr()
    for v in 1:n_vertices
        for k in 1:n_cores
            if k != c[v]
                println(string("Adicionando x[", v, ",", k, "]"))
                add_to_expression!(trocas_de_cor, x[v,k])
            end
        end
    end

    @objective(modelo, Min, trocas_de_cor)

    # --- adicionando a restrição de uma unica cor por vertice ---
    for v in 1:n_vertices
        soma_cores_vertice = AffExpr()
        for k in 1:n_cores
            add_to_expression!(soma_cores_vertice, x[v,k])
        end
        @constraint(modelo, soma_cores_vertice == 1)
    end

    # --- adicionando as restrições de convexidade ---
    for k in 1:n_cores
        for p in 1:n_vertices-2
            for q in p+1:n_vertices-1
                for r in q+1:n_vertices
                    @constraint(modelo, x[p,k] - x[q,k] + x[r,k] <= 1)
                end
            end
        end
    end

    # pede para o solver resolver o modelo
    optimize!(modelo)

    # se encontrou solução ótima, imprime solução
    if termination_status(modelo) == MathOptInterface.OPTIMAL
        imprime_solucao(x, n_vertices, n_cores, c)
    else
        println()
        println_in_yellow(string("Erro: Solver não encontrou solução ótima. Status = ", termination_status(modelo)))
    end
end

function imprime_solucao(x, n_vertices, n_cores, c)
    total_trocas=0
    println()
    println(string("Coloração original:  ", c))

    c2 = Array{Int}(undef, n_vertices)
    for v in 1:n_vertices
        for k in 1:n_cores
            if value(x[v,k]) >= 1
                c2[v] = k
                if k != c[v]
                    total_trocas += 1
                end
            end
        end
    end

    println(string("Coloração escolhida: ", c2))
    println()
    println(string("Total de trocas: ", total_trocas))
end

function executa_teste()
    arq_instancia = "instancias/rand_20_3.txt"
    dados_entrada = le_dados_entrada(arq_instancia)
    print(dados_entrada)
    resolve_recoloracao_convexa(dados_entrada...)
end

executa_teste()
