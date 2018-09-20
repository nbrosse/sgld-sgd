using JLD
using Distributions
include("./models/control_variates_bpmf.jl")




"""
Calculate full grad log post at mode
"""
function grad_log_post(model::matrix_factorisation, tuning::sgd)
    println("Calculating full gradlogpost...")
    (dlogU, dlogV, dloga, dlogb) = dlogpostfull(model, model.U, model.V,
                                               model.a, model.b)
    tuning.ll_U = dlogU
    tuning.ll_V = dlogV
    tuning.ll_a = dloga
    tuning.ll_b = dlogb
end

"""
Storage object
"""
type Storage
    U::Array{Float64,3}              # Latent feature vector
    V::Array{Float64,3}              # Latent feature vector
    a::Array{Float64,2}              # User bias terms
    b::Array{Float64,2}              # Item bias terms
    dlogU::Array{Float64,3}
    dlogV::Array{Float64,3}
    dloga::Array{Float64,2}
    dlogb::Array{Float64,2}
    # rmse::Array{Float64,1}
end

function Storage(model)
    U = zeros((2, model.D, model.L))
    dlogU = zeros((2, model.D, model.L))
    V = zeros((2, model.D, model.M))
    dlogV = zeros((2, model.D, model.M))
    a = zeros(2, model.L)
    dloga = zeros(2, model.L)
    b = zeros(2, model.M)
    dlogb = zeros(2, model.M)
    # rmse = zeros(Int(n_iter / 10))
    Storage(U, V, a, b, dlogU, dlogV, dloga, dlogb)
end

function run_mcmc(model::matrix_factorisation, stepsize::Float64,
        subsize::Int64, n_iter::Int64, sgd_init::sgd, algo::String)
    model.U = sgd_init.U
    model.V = sgd_init.V
    model.a = sgd_init.a
    model.b = sgd_init.b
    update_Λ(model)
    # Generate objects and storage
    println("Run ", algo)
    println("Number of users: $(model.L)\tNumber of items: $(model.M)")
    tuning = sgld(model, subsize, stepsize)
    stored = Storage(model)
    if (algo=="SGLDFP")
        cv = control_variate(model, sgd_init)
    end
    chunk = round(Int64, 0.01*n_iter)
    ind = 1
    for tuning.iter in 1:n_iter
        # Print progress, check the chain hasn't diverged, store perplexity
        if ( tuning.iter % chunk == 0)
            println(ind)
            ind +=1
        end
        if ( ( sum(isnan(model.U)) > 0 ) | ( sum(isnan(model.V)) > 0 ) )
            print("\n")
            error("The chain has diverged")
        end
        # if ( tuning.iter % 10 == 0 )
        #     rmse_current = rmse(model)
        #     println("$(tuning.iter)\t$rmse_current")
        #     stored.rmse[Int(tuning.iter /10)] = rmse_current
        #     if ( ( sum(isnan(model.U)) > 0 ) | ( sum(isnan(model.U)) > 0 ) )
        #         print("\n")
        #         error("The chain has diverged")
        #     end
        # end
        # Update 1 iteration
        if (algo=="SGLDFP")
            cv_full_update(model, tuning, cv)
        else
            sgld_full_update(model, tuning)
        end
        # Update storage
        stored.U[1,:,:] += model.U / n_iter
        stored.U[2,:,:] += model.U .^ 2 / n_iter
        stored.dlogU[1,:,:] += model.dlogU / n_iter
        stored.dlogU[2,:,:] += model.dlogU .^2 / n_iter
        stored.V[1,:,:] += model.V / n_iter
        stored.V[2,:,:] += model.V .^2 / n_iter
        stored.dlogV[1,:,:] += model.dlogV / n_iter
        stored.dlogV[2,:,:] += model.dlogV .^2 / n_iter
        stored.a[1,:] += model.a / n_iter
        stored.a[2,:] += model.a .^2 / n_iter
        stored.dloga[1,:] += model.dloga / n_iter
        stored.dloga[2,:] += model.dloga .^2 / n_iter
        stored.b[1,:] += model.b / n_iter
        stored.b[2,:] += model.b .^2 / n_iter
        stored.dlogb[1,:] += model.dlogb / n_iter
        stored.dlogb[2,:] += model.dlogb .^2 / n_iter
    end
    if (algo=="SGLDFP")
        return( cv, stored )
    else
        return( stored )
    end
end

function run_sgd(model::matrix_factorisation, stepsize::Float64, subsize::Int64, n_iter::Int64)
    tuning = sgd(model, subsize, stepsize)
    for tuning.iter in 1:n_iter
        if tuning.iter % 10 == 0
            rmse_current = rmse(model)
            println("$(tuning.iter)\t$rmse_current")
            # open("rmse_out/$(model.L)/sgd-1.log", "a") do f
            #     write(f, "$(tuning.iter)\t$rmse_current\n")
            # end
        end
        sgd_update(model, tuning)
    end
    grad_log_post(model, tuning)
    return( tuning )
end

# function save_stored(stored::Storage, stepsize::Float64)
#     mkpath("zv")
#     save("zv/storage-$stepsize.jld", "storage", stored)
# end
