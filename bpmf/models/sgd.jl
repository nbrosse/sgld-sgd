include("matrix_factorisation.jl")

"""
Container for sgd parameters
"""
type sgd
    subsize::Int64                  # Minibatch size
    iter::Int64
    subsample::Array{ Int64, 1 }
    ϵ_U::Float64             # Stepsize tuning constants
    G_U::Array{Float64,2}
    ϵ_V::Float64             # Stepsize tuning constants
    G_V::Array{Float64,2}
    ϵ_a::Float64             # Stepsize tuning constants
    G_a::Array{Float64,1}
    ϵ_b::Float64             # Stepsize tuning constants
    G_b::Array{Float64,1}
    ll_U::Array{ Float64, 2 }
    ll_V::Array{ Float64, 2 }
    ll_a::Array{ Float64, 1 }
    ll_b::Array{ Float64, 1 }
    U::Array{ Float64, 2 }
    V::Array{ Float64, 2 }
    a::Array{ Float64, 1 }
    b::Array{ Float64, 1 }
end

function sgd( model::matrix_factorisation, subsize::Int64, opt_stepsize::Float64 )
    subsample = sample( 1:model.N, subsize )
    ϵ_U = opt_stepsize
    ϵ_V = opt_stepsize
    ϵ_a = opt_stepsize
    ϵ_b = opt_stepsize
    G_U = ones( size(model.U) )
    G_V = ones( size(model.V) )
    G_a = ones( size(model.a) )
    G_b = ones( size(model.b) )
    ll_U = zeros( size(model.U) )
    ll_V = zeros( size(model.V) )
    ll_a = ones( size(model.a) )
    ll_b = ones( size(model.b) )
    U = zeros( size(model.U) )
    V = zeros( size(model.V) )
    a = zeros( size(model.a) )
    b = zeros( size(model.b) )
    sgd( subsize, 1, subsample, ϵ_U, G_U, ϵ_V, G_V, ϵ_a, G_a, ϵ_b, G_b, ll_U, ll_V, ll_a, ll_b,
            U, V, a, b )
end

function dlogpostΛ( model::matrix_factorisation )
    dlogΛ_U = zeros(model.D)
    dlogΛ_V = zeros(model.D)
    for d in 1:model.D
        dlogΛ_U[d] = ( model.α + model.L/2 - 1 )/model.Λ_U[d] - model.β - 1/2*sum( model.U[d,:].^2 )
        dlogΛ_V[d] = ( model.α + model.M/2 - 1 )/model.Λ_U[d] - model.β - 1/2*sum( model.V[d,:].^2 )
    end
    return ( dlogΛ_U, dlogΛ_V )
end

"""
Update one step of stochastic gradient descent
"""
function sgd_update( model::matrix_factorisation, tuning::sgd )
    tuning.subsample = sample( 1:model.N, tuning.subsize )
    ζ = 1 / ( 1 + tuning.iter )    
    ( dlogU, dlogV, dloga, dlogb ) = dlogpost( model, tuning.subsample, model.U, model.V, 
                                               model.a, model.b )
    tuning.G_U += ζ*( dlogU.^2 - tuning.G_U )
    model.U += tuning.ϵ_U / 2 * dlogU ./ sqrt( tuning.G_U )
    tuning.G_V += ζ*( dlogV.^2 - tuning.G_V )
    model.V += tuning.ϵ_V / 2 * dlogV ./ sqrt( tuning.G_V )
    tuning.G_a += ζ*( dloga.^2 - tuning.G_a )
    model.a += tuning.ϵ_a / 2 * dloga ./ sqrt( tuning.G_a )
    tuning.G_b += ζ*( dlogb.^2 - tuning.G_b )
    model.b += tuning.ϵ_b / 2 * dlogb ./ sqrt( tuning.G_b )
    tuning.U = model.U
    tuning.V = model.V
    tuning.a = model.a
    tuning.b = model.b
    if tuning.iter % 10 == 0
        update_Λ( model )
    end
end