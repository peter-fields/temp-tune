"""
define and generate trainig data from an instance of the toy model with given values of number of low- and high-energy states and energy gap.
"""
etrues(Δ,nlevels, nexcited, nground) = begin
    i=1
    # nstates = isodd(nlevels) ? (nlevels)*2 : (nlevels+1)*2
    energies=[]
    labels=[]
    gap_decay=zeros(nlevels)
    gap_decay[1]=0.
    for k in 2:nlevels
        gap_decay[k] = gap_decay[k-1]+1/(k-1)
        # gap_decay[k]=1
    end
    nn=[nground, nexcited]
    for k in 1:nlevels
        append!(energies, ones(nn[k]) *Δ*gap_decay[k] )
        append!(labels,   ones(nn[k]) *(k) )
    end
    energies, labels
end
function get_probs_2( energies )
    probs_true=exp.(-energies)
    probs_true./=sum(probs_true)
end
function get_freqs( sampless, lbls )
    freqs = Float64[]
    statess = collect(1:length(lbls))
    for i in statess
        append!(freqs, sum( sampless .== i ) )
    end
    freqs./=sum(freqs)
end
function get_samples( weightss, nsamps, nstates )
    samps_idx=sample( collect(1:nstates) , Weights(weightss) , nsamps )
end
function get_p_toy( Ls, Δ )
    pt = copy(Ls)
    pt = exp.(-Δ*pt)
    pt ./= sum(pt)
    return pt
end

"""
functions for fitting a model to an empirical distribution, returning estimate of level assignment vector, L̂, and estimate of the energy gap, Δ̂
"""

xlogx(x)= x != 0. ? x*log(x) : 0.

struct FitToyModelBuffer{T} 
    p_ñ_g::T
    ñ_g::T
    N_states::Real
    most_to_least_idx::AbstractVector{Integer}
    ground_lbls_current::T
    level_assignment_vector::T
    minlosses::T
    Deltas::T
    Lp::T
    function FitToyModelBuffer(p_emp::AbstractVector)
        N_s = length(p_emp)
        TT = eltype(p_emp)
        all_vecs = ntuple( i -> zeros(TT, N_s), 5 )
        return new{typeof(p_emp)}( zeros(TT, 1), zeros(TT, 1), N_s, zeros(Int64,N_s), all_vecs...)
    end
    function FitToyModelBuffer( N_states::Integer)
        TT = Float64
        all_vecs = ntuple( i -> zeros(TT, N_states), 5 )
        return new{typeof(all_vecs[1])}( zeros(TT, 1), zeros(TT, 1), N_states,
                            zeros(Int64,N_states), all_vecs...)
    end
end

function fit_n_ground_and_Δ_hard_constraint( p_emp, n_training_samps,
                                    fit_buffer::FitToyModelBuffer; 
                                    make_pfit=false,  regularize = true, λ=0)
    (; p_ñ_g, ñ_g, N_states, most_to_least_idx, 
            ground_lbls_current, level_assignment_vector, minlosses, Deltas, Lp ) = fit_buffer

    loss(q, n_g_) = -xlogx(q)-xlogx(1-q)+q*log( n_g_ / N_states )+(1-q)*log(1-(n_g_ /N_states))
    Δ̃(q, n_g_) = log( q * ( (N_states/n_g_)-1 ) / (1-q) )
    
    scramble_idx = StatsBase.sample(1:N_states, 
        N_states, replace=false)
    unscramble_idx=sortperm(scramble_idx) 
    
    p_emp = p_emp[scramble_idx] # make sure no bias towards certain degenerate solutions
                                # that are introduced from ordering of states
    
    most_to_least_idx.=reverse(sortperm(p_emp))
    
    L = level_assignment_vector
    L .= 1. #init all states as excited
        
    ground_lbls_current .= eltype(L).( .! Bool.(L) )
    
    for (i,k) in enumerate(most_to_least_idx[1:end]) 
        #all states in ground state gives log(0/0) for energy gap. send to infinity for stability
        i == N_states && ( Deltas[k]=Inf ; minlosses[k]=Inf ; continue )
        L[k]=0.
        Lp[k] = dot(p_emp ,L)
        ground_lbls_current[k]=1.
        sum!(ñ_g, ground_lbls_current)
        p_ñ_g.=dot(ground_lbls_current, p_emp)
        p_ñ_g.=round.(p_ñ_g[],digits=5) # prevent overfloat
        
        # if p_ñ_g == 1, change it s.t. prob of seeing any excited stated is 1/(M+1)
        # this corresponds to a hard constraint on the energu function
        if regularize == true
            p_ñ_g .= p_ñ_g[] == 1. ? 1 - (1/(n_training_samps+λ)) : p_ñ_g[]
        end
        
        Deltas[k] = Δ̃(p_ñ_g[], ñ_g[])
        
        minlosses[k] = loss(p_ñ_g[], ñ_g[])
    end
    
    # idxx=Bool.(isnan.(Deltas) + isinf.(Deltas))
    # # idxx=[]
    minl_orig = copy(minlosses)
    # minlosses[idxx].=Inf
    minlosses_m_to_l = minlosses[most_to_least_idx]

    n̂_g =findmin( minlosses_m_to_l )[2]
    
    Δ̂ = Deltas[most_to_least_idx][n̂_g]
    L .=1.
    L[most_to_least_idx[1:n̂_g]].=0.
    
    if Δ̂ < 0.
         # a negative energy is the same as swapping level assignments 
        Δ̂ = -Δ̂
        eltype_=eltype(L)
        L = Bool.(L)
        L .= .! L
        L = eltype_.(L)
    end
    
    
    # unscramble 
    minlosses=minlosses[unscramble_idx]
    L = L[unscramble_idx]
    Lp = Lp[unscramble_idx]

    pfit=nothing
    if make_pfit == true
        pfit = copy(L)
        if Δ̂ != Inf
            pfit = exp.(-Δ̂*pfit)
        elseif Δ̂ == Inf
            pfit.=0.
            pfit[.! (Bool.(L))].=1/n̂_g
        end
        pfit ./= sum(pfit)
    end
    
    return minlosses, Δ̂, L, pfit, Deltas, Lp, minlosses_m_to_l
end
    
fit_n_ground_and_Δ_hard_constraint( p_emp, n_training_samps; 
    make_pfit=false, regularize=true, λ=0) = fit_n_ground_and_Δ_hard_constraint( p_emp, n_training_samps, 
                                                                            FitToyModelBuffer(p_emp); 
                                                                            make_pfit=make_pfit,
                                                                            regularize=regularize,
                                                                            λ=λ)

"""
functions for sweeping over many replicates of sampling and fitting from ground truth distribution
"""


function run_sweeps_for_1_value_of_Ns_and_ng(Deltas, Ms, nlevels,nexcited,nground,nreps, N_states)
    df_sweep_Delta_M = DataFrame(Δ=Float64[], M=Float64[], n_g = Float64[], N_s = Float64[],
                                        Δ̂=Float64[], n̂_g=Float64[], 
                                        LL=Float64[], tau_star=Float64[], tau_prime=Float64[],
                                        L_fit=[], L_true=[])
    fit_buffer = FitToyModelBuffer(N_states)
    
    for Delta in Deltas
        for nsamps in Ms
            #define true model
            ees, lbls = etrues(Delta,nlevels,nexcited,nground)
            true_probs=get_probs_2( ees )
            # σ_true = [ zeros(nground) ; ones(nexcited) ]
            # do nreps replicates of sampling and fitting model
            for r in 1:nreps
                sampless=get_samples(true_probs, nsamps, N_states );
                freqs = get_freqs(sampless, lbls);
                minlosses, Δ̂, Li, pfit, Δs, Lp, minl_orig =fit_n_ground_and_Δ_hard_constraint( freqs, 
                                                            nsamps, fit_buffer;
                                                            make_pfit=false,
                                                            regularize=true);
                # ground_found_idx = (σ_true .== 0) .&  (Li .== 0)
                n_hat_g = sum(Float64.(Li .== 0))
                L_true=lbls.-1.
                L_fit=Li
                LL=L_true'*L_fit
                
                tau_starr=get_tau_star(nsamps, n_hat_g, Δ̂, N_states, nground, nexcited, Delta, LL )
                tau_primee=get_tau_prime(nsamps, n_hat_g, Δ̂, N_states, nground, nexcited, Delta, LL )
                
                push!(df_sweep_Delta_M, [Delta,nsamps,nground,N_states,
                        Δ̂, n_hat_g, LL, tau_starr, tau_primee, [L_fit], [L_true]] )
            end
        end
    end 
    return df_sweep_Delta_M
end

function get_mean_fit_params(Deltas, Ms, nlevels,nexcited,nground,nreps, df_sweep_Delta_M, N_states)
    df_mean_fits = DataFrame(Δ=Float64[], M=Float64[], n_g = Float64[], N_s = Float64[],
    Δ̂_mean=Float64[] , Δ̂_std=Float64[], 
    n̂_g_mean=Float64[], n̂_g_std=Float64[],
    LL_mean=Float64[], LL_std=Float64[], 
    tau_star_mean=Float64[], tau_star_std=[], 
    tau_prime_mean=Float64[], tau_prime_std=[])

    for Delta in Deltas
        for nsamps in Ms
            function filt_func( Δd, MM, n_gg, N_ss )
                ((Δd == Delta) && (MM == nsamps ) && (n_gg == nground) && (N_ss == N_states))
            end
            df_temp = filter( [:Δ, :M, :n_g, :N_s] => filt_func, df_sweep_Delta_M )
            # symbolss = [ :Δ̂, :n̂_g, :LL, :tau_star, :tau_prime ]
            # for ss in symbolss
            delta_hat_mean = mean(df_temp[!, :Δ̂])
            delta_hat_std = std( df_temp[!, :Δ̂])
            n_g_hat_mean = mean(df_temp[!, :n̂_g])
            n_g_hat_std = std(df_temp[!, :n̂_g])
            LL_mean = mean(df_temp[!, :LL])
            LL_std = std(df_temp[!, :LL])
            tau_star_mean = mean(df_temp[!, :tau_star])
            tau_star_std = std(df_temp[!, :tau_star])
            tau_prime_mean = mean(df_temp[!, :tau_prime])
            tau_prime_std = std(df_temp[!, :tau_prime])
            
            push!( df_mean_fits, [Delta,nsamps,nground,
                    N_states, 
                    delta_hat_mean , delta_hat_std, 
                    n_g_hat_mean, n_g_hat_std,
                    LL_mean, LL_std, 
                    tau_star_mean, tau_star_std, 
                    tau_prime_mean, tau_prime_std ] )
        end
    end

    return df_mean_fits
end

function make_arrays_for_contour(df_mean_fits, Deltas, nground, N_states)
    all_mean_n_g_hat, all_mean_delta_hat, all_mean_LL, all_tau_star, all_tau_prime,
    all_std_n_g_hat , all_std_delta_hat , all_std_LL, all_tau_star_std, all_tau_prime_std = ntuple(i->zeros(length(Ms), length(Deltas)), 10)
    for (k, D) in enumerate(Deltas)
        function filt_func( Δd, n_gg, N_ss )
            ((Δd == D) && (n_gg == nground) && (N_ss == N_states))
        end
        df_temp = filter( [:Δ, :n_g, :N_s] => filt_func, df_mean_fits )
        
        all_mean_delta_hat[:,k].=df_temp[!,:Δ̂_mean]
        all_std_delta_hat[:,k].=df_temp[!,:Δ̂_std]
        
        all_mean_n_g_hat[:,k] .=df_temp[!,:n̂_g_mean]
        all_std_n_g_hat[:,k]  .=df_temp[!,:n̂_g_std]
        
        all_mean_LL[:,k] .=df_temp[!,:LL_mean]
        all_std_LL[:,k]  .=df_temp[!,:LL_std]
        
        all_tau_star[:,k] .=df_temp[!,:tau_star_mean]
        all_tau_star_std[:,k] .=df_temp[!,:tau_star_std]
        
        all_tau_prime[:,k].=df_temp[!,:tau_prime_mean]
        all_tau_prime_std[:,k].=df_temp[!,:tau_prime_std]
        
    end
    
    return all_mean_n_g_hat, all_mean_delta_hat, all_mean_LL, all_tau_star, all_tau_prime, all_std_n_g_hat , all_std_delta_hat , all_std_LL, all_tau_star_std, all_tau_prime_std
end

function save_contour_fig(all_mean_delta_hat, all_mean_n_g_hat, Ms, tau_stars, tau_primes,
                                N_states, nground, nexcited, Deltas, dirstring)
    fig, ax = subplots(2,2, figsize=(9,7), sharey=true, sharex=true)
    ax=ax[:]

    epss = 0.9
    
    tau_stars = Matrix(tau_stars')
    tau_primes = Matrix(tau_primes')

    # @show size(all_mean_delta_hat)
    
    ncontours = 100
    
    cc=ax[1].contourf( log10.(Ms) , Deltas, all_mean_delta_hat' ./ Deltas, ncontours, cmap = "PRGn" ,vmin=1-epss,vmax=1+epss)
    cbb=colorbar(cc, orientation="vertical", location="right")
    ax[1].set_title(label=L"{\hat\Delta}/ {\Delta}" ,size=20)

    epss = 0.9
    cc=ax[2].contourf( log10.(Ms)[1:end] , Deltas, all_mean_n_g_hat'[:,1:end]./nground , ncontours, cmap = "PRGn" ,vmin=1-epss,vmax=1+epss)#, vmin=0.01,vmax=1. )
    cbb=colorbar(cc,orientation="vertical", location="right")
    ax[2].set_title(L"{\hat n_g}/{n_g}" , fontsize=15)

    epss=0.9
    # tau_stars[tau_stars.<0].=NaN
    # tau_stars[tau_stars.>5].=NaN
    cc=ax[3].contourf( (log10.(Ms)) , Deltas, tau_stars  , ncontours, cmap = "bwr", vmin=1-epss,vmax=1+epss)
    cbb=colorbar(cc, orientation="vertical", location="right")
    clns=ax[3].contour( (log10.(Ms)) , Deltas, round.(tau_stars,digits=3)  , [1.0], linewidths=0)
    coll=clns.collections[]
    
    x,y = get_phase_line(coll.get_paths())
    coll.set_paths(Vector{PyPlot.PyObject}())
    phase_line = (x,y)
    ax[3].plot(x,y)

    # @show clns.lines

    
    ax[3].set_title("opt τ (for reverse "*L"D_{KL}"*")" , size=13)

    # function ff(x)
    #     2. *x +1.7 
    # end
    # ax[3].plot( log10.(Ms)[75:105], ff.(log10.(Ms))[75:105], "k" )

    # ax[3].hlines((nexcited/nground)*exp(-0.15), ax[3].get_xlim()...)
    # ax[3].vlines(log10(nground+nexcited*exp(-0.15)), ax[3].get_ylim()...)

    ax[3].plot(x,y)

    for a in ax
        a.set_xlim(0.57,4.450002630173186)
        a.set_ylim(minimum(Deltas), maximum(Deltas))
        a.set_ylabel("Δ", fontsize=20, rotation ="horizontal", labelpad=20)
    end
    ax[2].set_xlabel("log(n training samples)", fontsize=10)

    ax[4].set_xlabel("log(n training samples)", fontsize=10)

    ax[1].set_ylabel("Δ", fontsize=20, rotation ="horizontal", labelpad=20)
    ax[2].set_ylabel("Δ", fontsize=20, rotation ="horizontal", labelpad=20)

    epss=2
    # tau_primes[tau_primes.<0].=NaN
    # tau_primes[tau_primes.>5].=NaN
    cc=ax[4].contourf( log10.(Ms) , Deltas, tau_primes , ncontours, cmap = "bwr", vmin=1-epss,vmax=1+epss)
    cbb=colorbar(cc, orientation="vertical", location="right")
    ax[4].set_title("opt τ (for forward "*L"D_{KL}"*")" , size=13)
    
    fig.suptitle("Ns=$N_states nground=$nground")
    
    savefig(projectdir()*"/temp_analysis/"*dirstring*"/Nstates=$(N_states)_nground=$nground.png")
    
    return fig, phase_line
end

# function get_tau_stars(Ms, all_mean_n_g_hat, all_mean_delta_hat,
#                                 N_states, nground, nexcited, Deltas)
#         #this assumes n_g_hat actually has n_g_hat/n_g_true and same for delt hat
#     denomm = (N_states .- (nground.*all_mean_n_g_hat')) .* exp.(Deltas) ./nground
#     numm = 1 .- all_mean_n_g_hat'
#     tau_stars = (all_mean_delta_hat'.*Deltas) ./ (Deltas .+ log.(1 .+ (numm./denomm)) )
#     return tau_stars
# end
# function get_tau_primes(Ms, all_mean_n_g_hat, all_mean_delta_hat,
#                                 N_states, nground, nexcited, Deltas)
#     #this assumes n_g_hat actually has n_g_hat/n_g_true and same for delt hat
#     denomm = (N_states .-nground.*all_mean_n_g_hat') 
#     numm =nexcited.+ exp.(Deltas) .* ( nground .- nground.*all_mean_n_g_hat' )
#     tau_primes = (all_mean_delta_hat'.*Deltas) ./ (Deltas .- log.( numm./denomm ) )
#     return tau_primes
# end

function get_tau_star(Ms, n_g_hat, delta_hat,
                                N_states, nground, nexcited, Delta, LL)
    n_e_h = ( N_states - n_g_hat )
    
    numm=n_e_h*n_g_hat
    denomm=LL*n_e_h - nexcited*n_e_h+LL*n_g_hat
    
#     numm = n_e_h    
#     denomm = (LL + (n_e_h / n_g_hat) * (LL - nexcited) )
    
    return (numm/denomm) *(delta_hat / Delta)
end
function get_tau_prime(Ms, n_g_hat, delta_hat,
                            N_states, nground, nexcited, Delta, LL)
    n_e_h = (N_states - n_g_hat)
    
    denomm = n_e_h*(n_g_hat + (nexcited-LL)*(exp(-Delta) - 1))
    numm = n_g_hat*(LL + exp(Delta) * ( nexcited - LL + nground - n_g_hat ))
    
    tau_prime = (delta_hat) / (Delta - log( numm/denomm ) )
    return tau_prime
end
function run_sweep_constant_N_states_vary_nground(N_states,fraction_nground::Vector{T},
                                                    Ms, Deltas, dirstring,
                                                    nlevels, nreps ) where T<:Any

    df_mean_fits_all_Ns_ng = DataFrame( N_states=[], nground=[], Ms=[], Deltas=[], nreps=[],
                                        all_mean_n_g_hat=[], 
                                        all_mean_delta_hat=[], 
                                        all_mean_LL=[], 
                                        all_tau_star=[], 
                                        all_tau_prime=[], 
                                        all_std_n_g_hat=[], 
                                        all_std_delta_hat =[], 
                                        all_std_LL =[],
                                        all_tau_star_std=[], 
                                        all_tau_prime_std=[],
                                        phase_line=[] ) 

    for frac_ng in fraction_nground
        @printf "%f\n" frac_ng

        nground = Int(round(frac_ng*N_states))

        nexcited = N_states - nground
        
        fnamee="Nstates=$(N_states)_nground=$nground.jld2"
        fpathh=joinpath(projectdir(),"temp_analysis",dirstring, fnamee)
        
        if isfile(fpathh)
            @printf "%s has already been run. skipping.\n" fpathh
            temp_dict=load( fpathh )
            push!( df_mean_fits_all_Ns_ng, temp_dict["sweep_data"])
            fig, phase_line = save_contour_fig( df_mean_fits_all_Ns_ng.all_mean_delta_hat[end][], 
                                    df_mean_fits_all_Ns_ng.all_mean_n_g_hat[end][], 
                                    Ms, df_mean_fits_all_Ns_ng.all_tau_star[end][], 
                                    df_mean_fits_all_Ns_ng.all_tau_prime[end][],
                                    N_states, 
                                    nground, nexcited, 
                                    Deltas,
                                    dirstring)
            df_mean_fits_all_Ns_ng.phase_line[end]=[phase_line]
            plt.close(fig)
            continue
        end

        

        df_sweep_Delta_M=run_sweeps_for_1_value_of_Ns_and_ng(Deltas, Ms, nlevels,nexcited,nground,nreps,N_states)

        @printf "sweeps done\n"

        df_mean_fits = get_mean_fit_params(Deltas, Ms, nlevels,nexcited,nground,nreps, df_sweep_Delta_M, N_states)

        all_mean_n_g_hat, all_mean_delta_hat, all_mean_LL, all_tau_star, all_tau_prime, all_std_n_g_hat , all_std_delta_hat , all_std_LL,all_tau_star_std, all_tau_prime_std = make_arrays_for_contour(df_mean_fits, Deltas, nground, N_states)

        fig, phase_line = save_contour_fig(all_mean_delta_hat, all_mean_n_g_hat, 
                                    Ms, all_tau_star, all_tau_prime,
                                    N_states, nground, nexcited, Deltas,
                                    dirstring)
        plt.close(fig)

        topushh = [ N_states, nground, [Ms], [Deltas], nreps,
                [all_mean_n_g_hat], [all_mean_delta_hat], [all_mean_LL], 
                [all_tau_star], [all_tau_prime], 
                [all_std_n_g_hat] , [all_std_delta_hat] , [all_std_LL] ,
                [all_tau_star_std], [all_tau_prime_std],
                [phase_line] ]

        push!( df_mean_fits_all_Ns_ng, topushh)

        fnamee="Nstates=$(N_states)_nground=$nground.jld2"
        jldsave( joinpath(projectdir(),"temp_analysis",dirstring, fnamee), 
            sweep_data=topushh )

        # fname_raw_data="Nstates=$(N_states)_nground=$(nground)_raw_data.csv"
        # CSV.write(joinpath(projectdir(),"temp_analysis",dirstring, fname_raw_data), df_sweep_Delta_M)
    end
    return df_mean_fits_all_Ns_ng
end
    
"""
other functions
"""

function get_dkl( probs1 , probs2 )
    xlogx(x)= x == 0 ? 0 : x*log(x)
    temp_vec=  xlogx.( probs1 ) .- probs1 .* log.( probs2 )
    return sum(temp_vec)
end