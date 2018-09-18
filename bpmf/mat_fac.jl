using Iterators
using StatsBase
using JLD
include("simulate.jl")

function truncate_data(n_users::Int64, train::Array{Int64,2}, test::Array{Int64,2})
    is_user_in = train[:,1] .<= n_users
    train = train[is_user_in,:]
    is_user_in = test[:,1] .<= n_users
    test = test[is_user_in,:]
    return(train, test)
end

function truncate_data(n_users::Int64, n_items::Int64, train::Array{Int64,2}, test::Array{Int64,2})
    is_user_in = train[:,1] .<= n_users
    train = train[is_user_in,:]
    is_item_in = train[:,2] .<= n_items
    train = train[is_item_in,:]
    is_user_in = test[:,1] .<= n_users
    test = test[is_user_in,:]
    is_item_in = test[:,2] .<= n_items
    test = test[is_item_in,:]
    @assert length(unique(train[:,1])) == n_users
    return(train, test)
end

# Latest version on github 
# index = 1
stepsize_list = Dict(10^2 => 5e-7,
                      5 * 10^2 => 1e-6,
                      NaN =>  1e-6);
sgd_step_list = Dict(10^2 => 1e-2,
                      5 * 10^2 => 1e-2,
                      NaN => 1e-2);
# seed_list = collect(1:5)
# list_size = length( seed_list )
# n_user_list = [10^2 5*10^2 NaN]

index = parse(Int64, ARGS[1]) # 1, 2, 3
# index = 1;
algo = ARGS[2] # 'sgld' or 'sgldfp'

# sgd_step = 3e-4
# stepsize = 1e-5
batchsize_sgd = 5000; # 5000
batchsize = 500; # 5000
# number_iter_sgd = 2 * 10^3;
# number_iter = 2 * 10^3;
n_user_list = [10^2 5 * 10^2 NaN];
n_users = n_user_list[index];
# sgd_step = sgd_step_list[n_users]
# stepsize = stepsize_list[n_users]


# seed_current = 1;
# srand(seed_current);
train = readdlm("data/u1.base");
test = readdlm("data/u1.test");
train = round(Int64, train);
test = round(Int64, test);
if !isnan(n_users)
    n_users = round(Int64, n_users);
    (train, test) = truncate_data(n_users, train, test);
end
model = matrix_factorisation(train, test);

sgd_step = 1e-2;
stepsize =  1 / model.N;
number_iter_sgd = 5*10^4;
number_iter = round(Int64, 10^2 * model.N);


sgd_init = run_sgd(model, sgd_step, batchsize_sgd, number_iter_sgd)
if algo=="sgld"
    store = run_mcmc(model, stepsize, batchsize, number_iter, sgd_init, "SGLD")
else
    cv_fp, store = run_mcmc(model, stepsize, batchsize, number_iter, sgd_init, "SGLDFP")
end

# var_U = mean(var(store.U, [1]));
# var_V = mean(var(store.V, [1]));
# var_U_fp = mean(var(store_fp.U, [1]));
# var_V_fp = mean(var(store_fp.V, [1]));
# var_dlogU = mean(var(store.dlogU, [1]));
# var_dlogV = mean(var(store.dlogV, [1]));
# var_dlogU_fp = mean(var(store_fp.dlogU, [1]));
# var_dlogV_fp = mean(var(store_fp.dlogV, [1]));

save_dict = Dict("U" => store.U,
                 "V" => store.V,
                 "dlogU" => store.dlogU,
                 "dlogV" => store.dlogV,
                 "a" => store.a,
                 "dloga" => store.dloga,
                 "b" => store.b,
                 "dlogb" => store.dlogb);

save(string(algo, "_", index, ".jld"), save_dict)


