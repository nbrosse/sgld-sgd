using Iterators
include("simulate.jl")

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

# res = run_sgld( model, stepsize, sgd_step, 5000, 2*10^3)

sgd_step = 3e-4
stepsize = 1e-5
batchsize_sgd = 5000
batchsize = 5000
number_iter_sgd = 2*10^3
number_iter = 2*10^3
n_user_list = [10^2 5*10^2 NaN]
n_users = NaN


seed_current = 1
srand(seed_current)
train = readdlm( "data/u1.base" )
test = readdlm( "data/u1.test" )
train = round( Int64, train )
test = round( Int64, test )
if !isnan(n_users)
    n_users = round( Int64, n_users )
    ( train, test ) = truncate_data( n_users, train, test )
end
model = matrix_factorisation(train, test)
sgd_init = run_sgd(model, sgd_step, batchsize_sgd, number_iter_sgd)
store1 = run_mcmc(model, stepsize, batchsize, number_iter, sgd_init, "SGLD")
store2 = run_mcmc(model, stepsize, 500, number_iter, sgd_init, "SGLD")

# tp1 = reshape(mean(res_sgld.U, 1), (model.D, model.L))
# tp2 = reshape(std(res_sgld.U, 1), (model.D, model.L))

