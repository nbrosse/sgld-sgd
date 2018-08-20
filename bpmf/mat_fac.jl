using Iterators
include("simulate.jl")

function test_sgld( model::matrix_factorisation, stepsize::Float64, sgd_step::Float64, seed::Int64 )
    println( "Running SGLD matrix factorization with $(model.L) users" )
    # res = run_sgld( model, stepsize, sgd_step, 5000, 2*10^3)
    res = run_sgld( model, stepsize, sgd_step, 500, 2*10^4)
    return res
end

function test_lmc( model::matrix_factorisation, stepsize::Float64, sgd_step::Float64, seed::Int64 )
    println( "Running LMC matrix factorization with $(model.L) users" )
    res = run_lmc( model, stepsize, sgd_step, model.N, 2*10^3)
    return res
end

function test_cv( model::matrix_factorisation, stepsize::Float64, sgd_step::Float64, seed::Int64 )
    println( "Running SGLDFP matrix factorization with $(model.L) users" )
    res = run_cv( model, stepsize, sgd_step, 5000, 2*10^3)
    return res
end

function test_sgd( model::matrix_factorisation, stepsize::Float64, sgd_step::Float64, seed::Int64 )
    println( "Running SGD matrix factorization with $(model.L) users" )
    res = run_sgd_smallstep( model, stepsize, sgd_step, 5000, 2*10^3)
    return res
end

function truncate_data( n_users::Int64, train::Array{Int64,2}, test::Array{Int64,2} )
    is_user_in = train[:,1] .<= n_users
    train = train[is_user_in,:]
    is_user_in = test[:,1] .<= n_users
    test = test[is_user_in,:]
    return( train, test )
end

function truncate_data( n_users::Int64, n_items::Int64, train::Array{Int64,2}, test::Array{Int64,2} )
    is_user_in = train[:,1] .<= n_users
    train = train[is_user_in,:]
    is_item_in = train[:,2] .<= n_items
    train = train[is_item_in,:]
    is_user_in = test[:,1] .<= n_users
    test = test[is_user_in,:]
    is_item_in = test[:,2] .<= n_items
    test = test[is_item_in,:]
    @assert length( unique( train[:,1] ) ) == n_users
    return( train, test )
end

# Latest version on github 
# index = 1
# stepsize_list = Dict( 10^2 => 5e-7,
#                       5*10^2 => 1e-6,
#                       NaN =>  1e-6 )
# sgd_step_list = Dict( 10^2 => 1e-2,
#                       5*10^2 => 1e-2,
#                       NaN => 1e-2 )
# seed_list = collect(1:5)
# list_size = length( seed_list )
# n_user_list = [10^2 5*10^2 NaN]

# stepsize = 1e-6
# sgd_step = 3e-4
sgd_step = 3e-4
n_user_list = [10^2 5*10^2 NaN]
n_users = NaN

stepsize = 1e-5

# seed_current = seed_list[floor( Int64, ( index - 1 ) % list_size + 1 )]
seed_current = 1
# println( "Fitting with stepsize: $stepsize" )
train = readdlm( "data/datasets/u1.base" )
test = readdlm( "data/datasets/u1.test" )
train = round( Int64, train )
test = round( Int64, test )
if !isnan(n_users)
    n_users = round( Int64, n_users )
    ( train, test ) = truncate_data( n_users, train, test )
end
srand(seed_current)
model = matrix_factorisation(train, test)
stored = test_sgld(model,stepsize,sgd_step,seed_current)
# model = matrix_factorisation(train, test)
# storedlmc = test_lmc(model,stepsize,sgd_step,seed_current)
# cvfp, storedfp = test_cv(model,stepsize,sgd_step,seed_current)
# model = matrix_factorisation(train, test)
# res_sgd = test_sgd(model,stepsize,sgd_step,seed_current)

res_sgld = stored
# res_cv = storedlmc

tp1 = reshape(mean(res_sgld.U, 1), (model.D, model.L))
tp2 = reshape(std(res_sgld.U, 1), (model.D, model.L))

res_sgld.rmse = res_sgld.rmse[res_sgld.rmse .!= 0]
# res_cv.rmse = res_cv.rmse[res_cv.rmse .!= 0]
# res_sgd.rmse = res_sgd.rmse[res_sgd.rmse .!= 0]
println("mean sgld rmse \t $(mean(res_sgld.rmse)) \t std sgld rmse $(std(res_sgld.rmse))")
# println("mean cv rmse \t $(mean(res_cv.rmse)) \t std cv rmse $(std(res_cv.rmse))")
# println("mean sgd rmse \t $(mean(res_sgd.rmse)) \t std sgd rmse $(std(res_sgd.rmse))")

# open("sortie.out", "w") do f
#     write(f, "############# D= 2 ##########")
#     write(f, "mean sgld rmse \t $(mean(res_sgld.rmse)) \t std sgld rmse $(std(res_sgld.rmse))\n")
#     write(f, "mean cv rmse \t $(mean(res_cv.rmse)) \t std cv rmse $(std(res_cv.rmse))\n")
#     write(f, "mean sgld U \t $(mean(res_sgld.U)) \t std sgld U $(std(res_sgld.U))\n")
#     write(f, "mean cv U \t $(mean(res_cv.U)) \t std cv U $(std(res_cv.U))\n")
# end

# tp1 = res_sgld.U[:,:,1]
# tp2 = res_cv.U[:,:,1]
