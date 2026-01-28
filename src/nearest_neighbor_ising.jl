"""
exact maximum likelihood
"""
function init_Jfit(N)
    Ĵ_start = randn(Float64,N,N)
    Ĵ_start=(Ĵ_start + Ĵ_start')/2
    Ĵ_start=Ĵ_start - diagm(diag(Ĵ_start))
    return Ĵ_start
end
struct ExactMLBuffer{T1, T2, T3, T4, T5}
    # N is number of spins in one state
    # M is 2^N
    model_moments::T1     # 1xNxN
    all_states::T2        # NxM
    all_states_energy::T3 # M
    exp_neg_energies::T3  # M
    σᵢσⱼ_all::T4          # MxNxN
    σᵢσⱼ_all_expE::T4     # MxNxN
    Z::T5                 # 1
    pemp::T3              # M
    all_state_idx_dict::Dict{Any,Any}
    function ExactMLBuffer( N::Integer, FloatType=Float64 )
        T = FloatType
        Z = zeros(T,1)
        model_moments = similar( Z,1, N, N )
        all_states = T.(Flux.onecold(enumerate_states(N, T), (-1,1)))
        M = size(all_states,2) #number of states
        all_states_energy = similar(Z,M)
        exp_neg_energies = similar(Z,M)
        σᵢσⱼ_all = similar(Z,M,N,N)
        diag_ones = diagm(ones(N))
        [ σᵢσⱼ_all[i,:,:] .= (x*x'-diag_ones) for (i,x) in enumerate(eachcol(all_states)) ];
        σᵢσⱼ_all_expE = similar(Z,M,N,N)
        pemp = similar(all_states_energy)
        all_state_idx_dict=get_all_state_idx_dict( all_states )
        return new{typeof(model_moments), typeof(all_states),typeof(all_states_energy), typeof(σᵢσⱼ_all), typeof(Z) }(
            model_moments,all_states,
            all_states_energy,exp_neg_energies, σᵢσⱼ_all,σᵢσⱼ_all_expE, Z, 
            pemp, all_state_idx_dict)  
    end
    function ExactMLBuffer( Jij::AbstractMatrix{T} ) where T<:AbstractFloat
        return ExactMLBuffer( size(Jij,1), eltype(Jij) )
    end
end
function get_2nd_moments_model( exact_ml_buffer, energy_buffer, Jij, N )
    (; model_moments, all_states, all_states_energy, 
        exp_neg_energies, σᵢσⱼ_all, σᵢσⱼ_all_expE, Z ) = exact_ml_buffer
    # energy2spin( all_states, reshape(Jij,N,N), energy_buffer )
    all_states_energy .= energy2spin( all_states, Jij, energy_buffer )
    @. exp_neg_energies = exp((-all_states_energy))
    σᵢσⱼ_all_expE .= σᵢσⱼ_all .* exp_neg_energies
    sum!(model_moments, σᵢσⱼ_all_expE)
    sum!(Z, exp_neg_energies )
    model_moments ./= Z
    return model_moments
end
function get_all_state_megabyte_count(FloatType, N)
    (sizeof(FloatType)*N*2^N)*10^-6
end
function enumerate_states(N, FloatType)
    # get all states of length N
    if get_all_state_megabyte_count( FloatType, N ) > 170.
         @error "too many spins. not enough memory for that bub"
    end
    s=BitArray(undef,N,2^N)
    for n in 0:2^N-1 
        scurr=collect(string(n,base=2,pad=N))
        s[:,n+1].=broadcast(x->parse(Bool,x) , scurr) 
    end
    return FloatType.(Flux.onehotbatch(s, (0,1)))
end
function get_2nd_moment_samples( samples )
    # samples should be N by M where M is number of samples
    rr = cov(samples, dims=2) + mean( samples, dims=2)*mean( samples, dims=2)'
    return rr - diagm(diag(rr))
end
function get_all_state_idx_dict( all_states )
    all_state_idx_dict=Dict( v=> i for (i,v) in enumerate(
                                        eachcol(all_states)))
end
function get_pemp( training_data, ml_bf_::ExactMLBuffer )
    (; all_states, pemp, all_state_idx_dict ) = ml_bf_
    # each collumn in both input matrices must reperesent a state
    𝒟 = training_data
    nsamps = size( 𝒟 , 2 )
    pemp .= eltype(pemp)(0)
    kcurr=0
    for samp_curr in eachcol(𝒟)
        # @show samp_curr
        kcurr = all_state_idx_dict[samp_curr]
        pemp[kcurr] += 1
    end
    
    normalize!(pemp, 1)
    return copy(pemp)
end

function get_pemp( training_data, all_states::AbstractMatrix )
    𝒟 = training_data
    n_states_total = size(all_states,2)
    pemp = zeros( n_states_total )
    pemp .= eltype(pemp)(0)
    
    for i in 1:n_states_total
        pemp[i] = sum( all(colll) for colll in eachcol( 𝒟 .== all_states[:,i]) )
    end
    
    normalize!(pemp, 1)
    return copy(pemp)
end

function ml_fit_Jij( Ĵ, 𝒟, fit_options, ml_exact_buffer ; level_indices = [] , silent=false) 
    (; learning_rate, relTol, max_iter, showevery) = fit_options
    (; pemp ) = ml_exact_buffer
    
    η=learning_rate
    Ĵ=Ĵ-diagm(diag(Ĵ)) # remove diagonal
    ∇Ĵ = similar(Ĵ)  #gradients
    N,M=size(𝒟)
    𝒟 = eltype(Ĵ).(𝒟)
    
    n_levels = length(level_indices)
    get_per_level = !isempty(level_indices)
    
    # for calculating gradients
    sample_corrs = get_2nd_moment_samples(𝒟)
    model_e_buffer = Energy2spinBuffer( ml_exact_buffer.all_states, Ĵ )
    model_corrs = similar(sample_corrs,1,N,N)
    
    # for calculating likelihoods
    e_buffer_𝒟 = Energy2spinBuffer( 𝒟, Ĵ )
    energies_𝒟 = similar( 𝒟, M )
    mean_energy_𝒟 = similar(𝒟,1)
    neg_log_like = similar(𝒟,1)
    neg_log_likes_all = []
    # dkl_per_lev = zeros(n_levels, 0)
    d_per_lev_curr = zeros(n_levels)
    p1_per_lev= get_per_level ? [zeros(sum(lil)) for lil in level_indices] : nothing
    p2_per_lev= get_per_level ? [zeros(sum(lil)) for lil in level_indices] : nothing
    
    if get_per_level 
        get_pemp( 𝒟, ml_exact_buffer )
    end
    # @show size(pemp)
    
    function update_grads!()
        model_corrs .= get_2nd_moments_model( ml_exact_buffer, model_e_buffer, Ĵ, N )
        ∇Ĵ .= (sample_corrs.-reshape(model_corrs,N,N))
    end
    function calc_likelihood()
        energies_𝒟 .= energy2spin(𝒟, Ĵ, e_buffer_𝒟 )
        mean!(mean_energy_𝒟,  energies_𝒟)
        neg_log_like .= log.(ml_exact_buffer.Z) .+ mean_energy_𝒟
    end
    function update_params!()
        lmul!(η,∇Ĵ)
        Ĵ .+= ∇Ĵ
        # Ĵ_aug[1,:,:] .+= Ĵ_aug[1,:,:]'
        # Ĵ_aug ./= 2
    end
    
    
    k=0
    stop_condition_not_met = true
    while stop_condition_not_met
        update_grads!()
        calc_likelihood()
        append!(neg_log_likes_all, neg_log_like)
        # if ( (get_per_level == true) && (k%showevery)==0 )
        #     get_dkl_per_level(pemp, ml_exact_buffer, level_indices; 
        #         dkls_per_lev=d_per_lev_curr,
        #         p1_per_lev=p1_per_lev,
        #         p2_per_lev=p2_per_lev)
        #     dkl_per_lev=matcolcat(dkl_per_lev, d_per_lev_curr)
        # end
        
        if ((k%showevery == 0) && (!silent))
            println("iter = ",k,", -likelihood = ",neg_log_like[1])
            println("   logZ=$(log.(ml_exact_buffer.Z))")
            println("   mean energy of train data = $(mean_energy_𝒟)")
            # println()
        end
        k == 0 && (k+=1; update_params!(); continue)
        if (abs(neg_log_likes_all[end] - neg_log_likes_all[end-1])/neg_log_likes_all[end-1]) < relTol
            stop_condition_not_met = false
            if get_per_level == true
                get_dkl_per_level(pemp, ml_exact_buffer, level_indices; 
                    dkls_per_lev=d_per_lev_curr,
                    p1_per_lev=p1_per_lev,
                    p2_per_lev=p2_per_lev)
            end
            
            (silent) && continue
            
            println("iter = ",k,", -likelihood = ",neg_log_like[1])
            println("relTol of ",relTol ," reached")
            println(); println();
            continue
        elseif k == max_iter
            stop_condition_not_met = false
            if get_per_level == true
                get_dkl_per_level(pemp, ml_exact_buffer, level_indices; 
                    dkls_per_lev=d_per_lev_curr,
                    p1_per_lev=p1_per_lev,
                    p2_per_lev=p2_per_lev)
            end
            (silent) && continue
            
            println("iter = ",k,", -likelihood = ",neg_log_like[1])
            println("maxIter of ",max_iter ," reached")
            println(); println(); 
            continue
        end
        update_params!()
        k+=1
    end
    if get_per_level == true
        return Ĵ, neg_log_likes_all, d_per_lev_curr
    else
        return Ĵ, neg_log_likes_all
    end
end
@with_kw struct FitOptions2spinExact{T}
    learning_rate::T = T.(0.01)
    relTol::T = T.(1e-7)
    max_iter::Int = 1000
    showevery::Int = 200
    function FitOptions2spinExact(J; learning_rate=0.01, 
            relTol=1e-7, max_iter=1000, showevery=200)
        new{eltype(J)}( learning_rate, relTol, max_iter, showevery)
    end
end
"""
energy function
"""
energy2spin( z, Jij ) = energy2spin( z, Jij, Energy2spinBuffer(z,Jij) )
function energy2spin( z::AbstractMatrix, Jij::AbstractMatrix, buffer )
    # z must be numspins X numsamples
    # Jij must be zero on the diagonal
    if diag(Jij) != zeros(eltype(Jij),size(Jij,1))
        @error "the diag elements of Jij must all be zero!"
    end
    (; J_times_σ, σ_J_σ, pre_E, E) = buffer
    mul!(J_times_σ, Jij, z)
    @. σ_J_σ = z * J_times_σ
    sum!( pre_E, σ_J_σ )
    T=eltype(pre_E)
    rmul!(pre_E, -1//2 )
    E .= vec( pre_E )
    return E
end
struct Energy2spinBuffer{VV,TT,GG}
    J_times_σ::VV # NxM
    σ_J_σ::VV     # NxM
    pre_E::TT     # 1xM
    E::GG         # M
    function Energy2spinBuffer( samples::AbstractMatrix, J::AbstractMatrix{T}  ) where T
        N,M = size( samples )
        E = zeros(T, M)
        J_times_σ = similar(E, N,M)
        σ_J_σ = similar(E,N,M)
        pre_E = similar(E,1,M)
        return new{typeof(J_times_σ),typeof(pre_E),typeof(E)}(J_times_σ, σ_J_σ, pre_E, E)
    end   
end
"""
init params for 2d NN
"""
function init_2dNN(m::Int, n::Int ; similarto=Float64[])
    # 2D nearest-neighbor ising model defined on m x n
    # lattice with periodic boundary conditions
    g=Graphs.SimpleGraphs.grid([m,n], periodic=true)    
    am=adjacency_matrix(g)
    J_ising = am
    # @show edges(g)
    q,N = (2, m*n) # q=2 spin-states
    θ = Pairwise(; q=q, N=N, similarto=similarto);
    Jtoy=eltype(θ).(diagm(ones(2)))
    J=zeros(eltype(θ), q,N,q,N)
    for i in axes(am,2), j in axes(am,1)
        if am[j,i] == 1
            J[:,j,:,i] .= Jtoy
        else
            J[:,j,:,i] .= zeros(eltype(θ),q,q)
        end
    end
    copyto!(θ.J, J)
    copyto!(θ.h, zeros(eltype(θ), 2,N))
    zerosum!(θ)
    return θ, eltype(θ).(Matrix(J_ising))
end
"""
exact sampler
"""
function get_true_prob(ml_exact_buffer, Jtrue)
    (; all_states, all_states_energy, exp_neg_energies) = ml_exact_buffer
    
    all_states_energy.=energy2spin(all_states, Jtrue)
    @. exp_neg_energies=exp(-all_states_energy)
        
    exp_neg_energies ./ sum(ml_exact_buffer.exp_neg_energies)
end
function drawsamples_exact(ml_exact_buffer, Jtrue, temp, nsamples)
    (; all_states, all_states_energy, exp_neg_energies) = ml_exact_buffer
    all_states_energy.=energy2spin(all_states, Jtrue./temp)
    @. exp_neg_energies=exp(-all_states_energy)
    all_states_idxs = collect(1:length(all_states_energy))
    samples_idxs = StatsBase.sample( 
        all_states_idxs,
        Weights( exp_neg_energies ./ sum(ml_exact_buffer.exp_neg_energies) ), 
        nsamples )
    return all_states[:,samples_idxs]
end
"""
exact entropy calculations
"""
function heat_capacities_exact( Jij, temp, ml_exact_buffer_for_C )
    (; all_states, all_states_energy, Z, exp_neg_energies ) = ml_exact_buffer_for_C
    all_states_energy .= energy2spin(all_states, Jij )
    exp_neg_energies .= exp.(-all_states_energy./temp)
    sum!(Z, exp_neg_energies)
    p = exp_neg_energies ./ Z
    E_mean = sum(p.*all_states_energy)
    ΔE² = (all_states_energy .- E_mean) .^2
    C = sum(p .* ΔE²) / (temp^2)
end
function entropy_exact( ml_exact_buffer, Jtrue , true_temp)
    p2=energy2spin(ml_exact_buffer.all_states, Jtrue./true_temp)
    p2=p2.-minimum(p2)
    p2=exp.(-p2)
    p2./=sum(p2)
    ent = -sum(p2.*log.(p2))
    return ent
end
function dkl_p_rand_with_p_true( ml_exact_buffer, Jtrue , true_temp )
    N=size(Jtrue,1)
    p2=energy2spin(ml_exact_buffer.all_states, Jtrue./true_temp)
    p2=p2.-minimum(p2)
    p2=exp.(-p2)
    p2./=sum(p2)
    Hrand=N*log(2) 
    return -Hrand - (1/(2^N))*sum( log.(p2) )
end
"""
misc funcs
"""
function remove_diag(A)
    A + diagm(fill(NaN, size(A,1)))
end
function Jij_to_θ( Jij )
    N = size(Jij, 2)
    θ = Pairwise(;q=2,N=N)
    J = zeros(eltype(θ),2,N,2,N)
    for i in axes(Jij,2), j in axes(Jij,1)
        if i != j
            # @show size(J)
            J[:,j,:,i] .= Jij[j,i]*diagm(ones(eltype(θ),2))
        end
    end
    copyto!(θ.J, J)
    copyto!(θ.h, zeros(eltype(θ), 2,N))
    zerosum!(θ)
end
function θ_to_Jij( θ )
    J = zeros(eltype(θ),θ.N,θ.N)
    for i in axes(J,2), j in axes(J,1)
        if i != j
            # @show size(J)
            J[j,i] = abs(θ.J[1,j,1,i]*2)
        end
    end
    return J
end
matcolcat(a, b) = reshape(append!(vec(a), vec(b)), size(a)[1:end-1]..., :)

function get_dkl( probs1 , probs2 )
    xlogx(x)= x == 0 ? 0 : x*log(x)
    temp_vec=  xlogx.( probs1 ) .- probs1 .* log.( probs2 )
    return sum(temp_vec)
end
function get_ent( probs1 )
     xlogx(x)= x == 0 ? 0 : x*log(x)
     sum(-xlogx.( probs1 ) )
end
function get_cross_ent( probs1 , probs2 )
    sum( - probs1 .* log.( probs2 ) )
end

    
function get_level_index_dict(J_true, ml_exact_buffer, e_buffer_all_states)
    (; all_states, all_states_energy, 
        exp_neg_energies,  Z ) = ml_exact_buffer
    all_states_energy .= energy2spin( all_states , J_true , e_buffer_all_states )
    e_level_vals = sort( unique( all_states_energy ))
    idx_dict = Vector{Any}(missing, length(e_level_vals))
    for (l, e_vall) in enumerate(e_level_vals)
        idx_dict[l] = all_states_energy .== e_vall
    end
    return idx_dict
end
function get_rev_dkl_per_level(pemp, ml_exact_buffer, level_index_dict)
    (; all_states, all_states_energy, 
        exp_neg_energies,  Z ) = ml_exact_buffer
    dkls_per_lev = zeros(length(level_index_dict))
    # exp_neg_energies ./= sum(exp_neg_energies)
    for l in 1:length(level_index_dict)
        g_idx = level_index_dict[l]
        dkls_per_lev[l] = get_dkl( exp_neg_energies[g_idx]./Z[] , pemp[g_idx])
    end
    return dkls_per_lev
end
function get_dkl_per_level(pemp, ml_exact_buffer, level_index_dict; 
                            dkls_per_lev=zeros(length(level_index_dict)),
                            p1_per_lev=[zeros(sum(lil)) for lil in level_index_dict],
                            p2_per_lev=[zeros(sum(lil)) for lil in level_index_dict])
    (; all_states, all_states_energy, 
        exp_neg_energies,  Z ) = ml_exact_buffer
    # dkls_per_lev .= zeros(length(level_index_dict))
    # exp_neg_energies ./= sum(exp_neg_energies)
    for l in 1:length(level_index_dict)
        # g_idx = 
        p1_per_lev[l] .= pemp[level_index_dict[l]]
        p2_per_lev[l] .= exp_neg_energies[level_index_dict[l]] ./ Z[]
        dkls_per_lev[l] = get_dkl( p1_per_lev[l] , p2_per_lev[l] )
    end
    return dkls_per_lev
end

