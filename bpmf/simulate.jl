using JLD
using Distributions
include("./models/control_variates_bpmf.jl")
include("./models/matrix_factorisation.jl")




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
    U::Array{Float64,2}              # Latent feature vector
    V::Array{Float64,2}              # Latent feature vector
    a::Array{Float64,1}              # User bias terms
    b::Array{Float64,1}              # Item bias terms
    # dlogU::Array{Float64,3}
    # dlogV::Array{Float64,3}
    # dloga::Array{Float64,2}
    # dlogb::Array{Float64,2}
    rmse::Array{Float64,1}
end

function Storage(model, n_iter)
    U = zeros((model.D, model.L))
    # dlogU = zeros((2, model.D, model.L))
    V = zeros((model.D, model.M))
    # dlogV = zeros((2, model.D, model.M))
    a = zeros(model.L)
    # dloga = zeros(2, model.L)
    b = zeros(model.M)
    # dlogb = zeros(2, model.M)
    rmse = zeros(Int(n_iter / 10))
    # Storage(U, V, a, b, dlogU, dlogV, dloga, dlogb)
    Storage(U, V, a, b, rmse)
end

function rmse( model::matrix_factorisation, stored::Storage )
    rmse = 0
    test_size = size( model.test, 1 )
    for i in 1:test_size
        ( user, item, rating ) = model.test[i,:]
        try
            rmse += ( rating - dot( stored.U[:,user], stored.V[:,item]) -
                      stored.a[user] - stored.b[item] )^2
        catch
            continue
        end
    end
    rmse = sqrt( rmse / test_size )
    return rmse
end

function run_mcmc(model::matrix_factorisation, stepsize::Float64,
        subsize::Int64, n_iter::Int64, sgd_init::sgd, algo::String, reinit_bool::Bool)
    model.U = sgd_init.U
    model.V = sgd_init.V
    model.a = sgd_init.a
    model.b = sgd_init.b
    update_Λ(model)
    if reinit_bool
        reinit(model)
    end
    # Generate objects and storage
    println("Run ", algo)
    println("Number of users: $(model.L)\tNumber of items: $(model.M)")
    if (algo=="SGD")
        tuning = sgd(model, subsize, stepsize)
    else
        tuning = sgld(model, subsize, stepsize)
    end
    stored = Storage(model, n_iter)
    if (algo=="SGLDFP")
        cv = control_variate(model, sgd_init)
    end
    # chunk = round(Int64, 0.01*n_iter)
    # ind = 1
    for tuning.iter in 1:n_iter
        # Print progress, check the chain hasn't diverged, store perplexity
        # if ( tuning.iter % chunk == 0)
        #     println(ind)
        #     ind +=1
        # end
        # if ( ( sum(isnan(model.U)) > 0 ) | ( sum(isnan(model.V)) > 0 ) )
        #     print("\n")
        #     error("The chain has diverged")
        # end
        if ( tuning.iter % 10 == 0 )
            rmse_current = rmse(model, stored)
            println("$(tuning.iter)\t$rmse_current")
            stored.rmse[Int(tuning.iter /10)] = rmse_current
            if ( ( sum(isnan(model.U)) > 0 ) | ( sum(isnan(model.U)) > 0 ) )
                print("\n")
                error("The chain has diverged")
            end
        end
        # Update 1 iteration
        if (algo=="SGLDFP")
            cv_full_update(model, tuning, cv)
        elseif (algo=="SGD")
            sgd_update(model, tuning)
        else
            sgld_full_update(model, tuning)
        end
        stored.U += (model.U - stored.U) / tuning.iter
        stored.V += (model.V - stored.V) / tuning.iter
        stored.a += (model.a - stored.a) / tuning.iter
        stored.b += (model.b - stored.b) / tuning.iter
    end
    # if (algo=="SGLDFP")
    #     return( cv, stored )
    # else
    return( stored )
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
