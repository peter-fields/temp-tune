
"""
code for finding optimal tau and running many replicates of experiments at different number of training samples, M, drawn from ground truth and at different ground truth temperatures T

note that in this code an old convention of T̂ = τ
"""
struct TauSweepOptions
    nreps::Int
    mu_vals_to_sweep
    true_temps_to_sweep::Vector 
    sweep_type::String
    m_by_n::Tuple{Int,Int}
    nsamples_to_sweep::Vector{Int}
    tau_to_sweep::Vector
    nsamples_to_sweep_per_mu::Tuple
    nsamples_to_sweep_all::Vector
    function TauSweepOptions(nreps, mu_vals_to_sweep, true_temps_to_sweep, sweep_type = "constant mu"; 
            m_by_n=(4,4), nsamples_to_sweep=[1000], tau_to_sweep=[] )
        if (sweep_type != "constant mu") && (sweep_type != "constant M")
            @error """ sweep type may only be "consant mu" or "constant M" """
        end
        if sweep_type == "constant M"
            mu_vals_to_sweep = fill("NA", length(nsamples_to_sweep))
        end
        if isempty(tau_to_sweep)
            tau_to_sweep=collect(range(0.2,5,length=70))
            (1 in tau_to_sweep) || insert!(tau_to_sweep, findall(tau_to_sweep.<1)[end]+1, 1.)
        end
        nsamples_to_sweep_per_mu, nsamples_to_sweep_all=get_nsamples_to_sweep_all( mu_vals_to_sweep, 
                                                                true_temps_to_sweep, sweep_type, m_by_n, nsamples_to_sweep )
        
        return new( nreps, mu_vals_to_sweep, true_temps_to_sweep, sweep_type, m_by_n, nsamples_to_sweep, tau_to_sweep, nsamples_to_sweep_per_mu, nsamples_to_sweep_all )
    end
end
function get_samples_at_oneT(T, θ, nsamples, sampler_opt_Tsamples; 
        mcmc_or_exact="exact")
    
    samples_at_oneT=Dict{Symbol,Any}()
    if mcmc_or_exact=="mcmc"
        traces, z = drawsamples( θ, T, sampler_opt_Tsamples );
        @show size(z)
        samples_at_oneT[:samples] = z
        samples_at_oneT[:sample_trace] = traces
        return samples_at_oneT
    elseif mcmc_or_exact=="exact"
        Jij = θ_to_Jij(θ)
        println("getting samples at T=$T")
        z=drawsamples_exact(ExactMLBuffer(Jij), Jij, T, sampler_opt_Tsamples.M)
        samples_at_oneT[:samples] = Flux.onehotbatch(z, (-1,1))
        println("done!")
        return samples_at_oneT
    end
end

function calcs_on_samples_at_oneT(temp,model,t_vs_t̂,
        fitmethod="exact",fit_options=FitOptions2spinExact() )
    # @show typeof(samples_at_oneT[:samples])
    samples_at_oneT = t_vs_t̂["T=$temp"]
    ml_exact_buffer=t_vs_t̂["ml_exact_buffer"]
    z = samples_at_oneT[:samples]
    # traces = samples_at_oneT[:sample_trace] 
    samples_at_oneT[:cov] = cov( Flux.onecold(z, (-1,1)), dims=2 )
    samples_at_oneT[:fit_model] = Dict{Symbol, Any}()
    if fitmethod == "exact"
        Jfit_start = init_Jfit(model.N)
        # ml_exact_buffer = ExactMLBuffer(Jfit_start)
        samples_at_oneT[:fit_model][:Ĵ], samples_at_oneT[:fit_model][:fit_neg_log_likes] = ml_fit_Jij( Jfit_start,
                                                                             Flux.onecold(z,(-1,1)) , 
                                                                                fit_options, ml_exact_buffer )
        # samples_at_oneT[:fit_model][:ml_exact_buffer] = ml_exact_buffer
    elseif fitmethod == "mean field"
        Jfit_start = init_Jfit(model.N)
        # ml_exact_buffer = ExactMLBuffer(Jfit_start)
        samples_at_oneT[:fit_model][:Ĵ] = mf_Jij( z )
        # samples_at_oneT[:fit_model][:ml_exact_buffer] = ml_exact_buffer
    end
    samples_at_oneT[:negloglike_trainingsamps_under_trueJ] = neg_avg_log_prob_samples(Flux.onecold(z,(-1,1)) , 
                                                            ml_exact_buffer.all_states,
                                                            θ_to_Jij(model)./temp )
    samples_at_oneT[:true_ent] = entropy_exact(ml_exact_buffer,θ_to_Jij(model),temp )
    
    return samples_at_oneT
end

# function calcs_on_samples_at_oneT_mod(temp,model,samples_at_oneT,fitmethod="exact",fit_options=FitOptions2spinExact())
#     # @show typeof(samples_at_oneT[:samples])
#     z = samples_at_oneT[:samples]
#     traces = samples_at_oneT[:sample_trace] 
#     samples_at_oneT[:negloglike] = neg_log_likelihood(z, model, temp)
#     samples_at_oneT[:true_ent] = entropy_onsager(temp, model.N)
#     samples_at_oneT[:cov] = cov( Flux.onecold(z, (-1,1)), dims=2 )
#     samples_at_oneT[:fit_model] = Dict{Symbol, Any}()
#     if fitmethod == "exact"
#         Jfit_start = init_Jfit(model.N)
#         ml_exact_buffer = ExactMLBuffer(Jfit_start)
#         samples_at_oneT[:fit_model][:Ĵ], samples_at_oneT[:fit_model][:fit_neg_log_likes] = ml_fit_Jij( Jfit_start,
#                                                                              z , fit_options, ml_exact_buffer )
#         samples_at_oneT[:fit_model][:ml_exact_buffer] = ml_exact_buffer
#     elseif fitmethod == "mean field"
#         Jfit_start = init_Jfit(model.N)
#         ml_exact_buffer = ExactMLBuffer(Jfit_start)
#         samples_at_oneT[:fit_model][:Ĵ] = mf_Jij( z )
#         samples_at_oneT[:fit_model][:ml_exact_buffer] = ml_exact_buffer
#     end
#     return samples_at_oneT
# end

function get_samples_all_T(; griddims=(3,3), nsamples=100,
                        temps_to_sweep=[2, 2.269, 2.5], sampler_opt_Tsamples,
                        mcmc_or_exact="mcmc")
    
    T_vs_T̂=Dict{String,Any}()
    T_vs_T̂["griddims"]=griddims
    θ,_=init_2dNN(griddims...)
    T_vs_T̂["θ"] = θ 
    T_vs_T̂["nsamples"] = nsamples
    T_vs_T̂["ml_exact_buffer"]= ExactMLBuffer(θ.N, eltype(θ))

    for t in temps_to_sweep
        T_vs_T̂["T=$t"]=get_samples_at_oneT(t,θ,nsamples, sampler_opt_Tsamples; 
            mcmc_or_exact=mcmc_or_exact)
    end
    return T_vs_T̂
end
function calcs_on_samples_all_T( T_vs_T̂; temps_to_sweep=[2, 2.269, 2.5], 
                                    fitmethod="exact", fit_options=FitOptions2spinExact(Float32) )
    θtrue = T_vs_T̂["θ"]
    for t in temps_to_sweep
        println("\n fitting model to data taken at T=$t")
        T_vs_T̂["T=$t"]=calcs_on_samples_at_oneT(t,θtrue,T_vs_T̂,
            fitmethod, fit_options)
    end
    return T_vs_T̂
end
function sweep_T̂(samples_at_oneT_dict, sampler_opt, ml_exact_buffer;
        mcmc_or_exact="mcmc",
        T̂_to_sweep = collect(range(0.05,1.95,length=15)))
    d=samples_at_oneT_dict
    # z=d[:samples]
    θ̂=Jij_to_θ(d[:fit_model][:Ĵ])
    # T̂_to_sweep = collect(range(0.05,2.0,length=15)) # 20% on either side of T
    d[:fit_model][:samples] = Dict{String, Any}()
    d[:fit_model][:T̂_to_sweep] = T̂_to_sweep
    if mcmc_or_exact == "exact"
        println("sweeping fit model tau from $(T̂_to_sweep[1]) to $(T̂_to_sweep[end])")
    end
    for t̂ in T̂_to_sweep
        if mcmc_or_exact == "mcmc"
            traces, z = drawsamples( θ̂, t̂, sampler_opt );
            d[:fit_model][:samples]["T̂=$t̂"] = z
        elseif mcmc_or_exact == "exact"
            # println("getting samples for T̂=$t̂")
            z=drawsamples_exact( ml_exact_buffer, d[:fit_model][:Ĵ],t̂,sampler_opt.M)
            d[:fit_model][:samples]["T̂=$t̂"] = Flux.onehotbatch( z , (-1,1) )
        end
    end
    println("done!")
end
function sweep_T̂_for_allT( T_vs_T̂_calcs, sampler_opt_forT̂_samples,
                            temps_to_sweep;
                            mcmc_or_exact="mcmc",
                            T̂_to_sweep = collect(range(0.05,1.95,length=15))  ) 
    for t in temps_to_sweep
        @printf "\n getting samples from fit model on samples taken at T=%f \n" t
        sweep_T̂( T_vs_T̂_calcs["T=$t"], sampler_opt_forT̂_samples, T_vs_T̂_calcs["ml_exact_buffer"]; 
            mcmc_or_exact=mcmc_or_exact, T̂_to_sweep=T̂_to_sweep )
    end
end
function calc_neg_log_likes( samples_oneT_dict , T , θgt, ml_exact_buffer) 
    neg_log_likes_under_groundtruth=zeros(
        size(samples_oneT_dict[:fit_model][:T̂_to_sweep]))
    neg_log_likes_under_groundtruth_approx=similar(neg_log_likes_under_groundtruth)
    # @show log_partition_onsager(1/T, θgt.N)
    # @show samples_oneT_dict[:fit_model][:T̂_to_sweep]
    for (i,t̂) in enumerate(samples_oneT_dict[:fit_model][:T̂_to_sweep])
        z=samples_oneT_dict[:fit_model][:samples]["T̂=$t̂"]
        neg_log_likes_under_groundtruth[i]=neg_avg_log_prob_exact(ml_exact_buffer.all_states,
                                                      samples_oneT_dict[:fit_model][:Ĵ]./t̂,  θ_to_Jij(θgt)./T)
        neg_log_likes_under_groundtruth_approx[i]=neg_avg_log_prob_samples( Flux.onecold(z, (-1,1)),
                                                            ml_exact_buffer.all_states,
                                                            θ_to_Jij(θgt)./T )
        # @show mean(energy(z, θgt))
    end
    samples_oneT_dict[:fit_model][:negloglikes_under_gt_approx]=neg_log_likes_under_groundtruth_approx
    samples_oneT_dict[:fit_model][:negloglikes_under_gt] = neg_log_likes_under_groundtruth
end

function calc_fit_model_true_ents_exact( samples_oneT_dict, ml_exact_buffer )
    ddd = samples_oneT_dict
    fit_models_true_ent = zeros(
        size(ddd[:fit_model][:T̂_to_sweep]))
    Jfit = ddd[:fit_model][:Ĵ]
    # ML_exact_buffer = ddd[:fit_model][:ml_exact_buffer]
    for (i,t̂) in enumerate(ddd[:fit_model][:T̂_to_sweep])
        fit_models_true_ent[i] = entropy_exact(ml_exact_buffer, Jfit, t̂ )
    end
    ddd[:fit_model][:fit_models_true_ent] = fit_models_true_ent
end

function calc_fitmodel_negloglikesT̂( samples_oneT_dict, θgt, T, ml_exact_buffer )
    ddd = samples_oneT_dict
    fitmodels_negloglikes = zeros(
        size(ddd[:fit_model][:T̂_to_sweep]))
    Jfit = ddd[:fit_model][:Ĵ]
    # ML_exact_buffer = ddd[:fit_model][:ml_exact_buffer]
    Jij_over_T_true = θ_to_Jij(θgt)./T 
    for (i,t̂) in enumerate(ddd[:fit_model][:T̂_to_sweep])
        fitmodels_negloglikes[i]=neg_avg_log_prob_exact(ml_exact_buffer.all_states, Jij_over_T_true , Jfit./t̂)
    end
    ddd[:fit_model][:fitmodels_negloglikesT̂] = fitmodels_negloglikes
end
    
# function calc_neg_log_likes_mod( samples_oneT_dict , T , θgt) 
#     neg_log_likes_under_groundtruth=zeros(
#         size(samples_oneT_dict[:fit_model][:T̂_to_sweep]))
#     # @show log_partition_onsager(1/T, θgt.N)
#     # @show samples_oneT_dict[:fit_model][:T̂_to_sweep]
#     for (i,t̂) in enumerate(samples_oneT_dict[:fit_model][:T̂_to_sweep])
#         z=samples_oneT_dict[:fit_model][:samples]["T̂=$t̂"]
#         neg_log_likes_under_groundtruth[i]=neg_log_likelihood(z,θgt,T)
#         # @show mean(energy(z, θgt))
#     end
#     samples_oneT_dict[:fit_model][:negloglikes_under_gt] = neg_log_likes_under_groundtruth
# end
function calc_all_negloglikesT̂( T_vs_T̂_calcs, temps_to_sweep )
    for t_gt in temps_to_sweep
        calc_neg_log_likes( T_vs_T̂_calcs["T=$t_gt"] , t_gt , T_vs_T̂_calcs["θ"],  
            T_vs_T̂_calcs["ml_exact_buffer"])
    end
end

function calc_all_fitmodeltrue_ents( T_vs_T̂_calcs, temps_to_sweep )
    for t_gt in temps_to_sweep
        calc_fit_model_true_ents_exact( T_vs_T̂_calcs["T=$t_gt"], T_vs_T̂_calcs["ml_exact_buffer"] )
    end
end

function calc_all_fitmodel_negloglikesT̂( T_vs_T̂_calcs, temps_to_sweep )
    for t_gt in temps_to_sweep
        calc_fitmodel_negloglikesT̂( T_vs_T̂_calcs["T=$t_gt"] , T_vs_T̂_calcs["θ"], t_gt, 
            T_vs_T̂_calcs["ml_exact_buffer"] )
    end
end
function calc_all_dkl_p_rand_p_true( T_vs_T̂_calcs, temps_to_sweep )
    ml_exact_buffer=T_vs_T̂_calcs["ml_exact_buffer"]
    Jtrue = θ_to_Jij(T_vs_T̂_calcs["θ"])
    for t_gt in temps_to_sweep
        oneT_dict=T_vs_T̂_calcs["T=$t_gt"]
        dkl_pr_pt = dkl_p_rand_with_p_true(ml_exact_buffer, Jtrue, t_gt)
        T_vs_T̂_calcs["T=$t_gt"][:dkl_pr_pt]=dkl_pr_pt
    end
end
function neg_avg_log_prob_exact( all_states, Jij_for_expectations, Jij  )
    p = energy2spin( all_states, Jij_for_expectations )
    p .= exp.(-p)
    p ./= sum(p)
    all_Es = energy2spin( all_states, Jij )
    logZ = exp.(-all_Es)
    logZ = log(sum(logZ))
    return logZ + sum( p .* all_Es )
end
function neg_avg_log_prob_samples( samples , all_states, Jij ) 
    p = energy2spin( all_states, Jij )
    p .= exp.(-p)
    logZ = log(sum(p))
    return logZ + mean( energy2spin( samples, Jij ) )
end
function neg_avg_log_prob_noZ( samples, Jij )
    return mean( energy2spin( samples, Jij ) )
end
function full_sweeps_and_fit( (m,n) , true_temps_to_sweep , n_training_samples,
                                sampler_opt_Tsamples,
                                sampler_opt_forT̂_samples;
                                mcmc_or_exact_Tsamples="mcmc",
                                fitmethod="exact", 
                                fit_options=FitOptions2spinExact(Float32),
                                mcmc_or_exact_T̂_samples="mcmc",
                                T̂_to_sweep = collect(range(0.05,2.0,length=15)) )
    m,n=(m,n)
    temps_to_sweep=true_temps_to_sweep
    nsamples=n_training_samples
    T_vs_T̂_=get_samples_all_T(; griddims=(m,n), nsamples=n_training_samples,
                        mcmc_or_exact=mcmc_or_exact_Tsamples,
                        temps_to_sweep=temps_to_sweep, sampler_opt_Tsamples)

    T_vs_T̂_calcs_=calcs_on_samples_all_T( T_vs_T̂_; temps_to_sweep=temps_to_sweep,
                                             fitmethod, fit_options )

    sweep_T̂_for_allT( T_vs_T̂_calcs_, sampler_opt_forT̂_samples,
                                temps_to_sweep,
                                mcmc_or_exact=mcmc_or_exact_T̂_samples,
                                T̂_to_sweep=T̂_to_sweep)

    calc_all_negloglikesT̂( T_vs_T̂_calcs_, temps_to_sweep )
    
    calc_all_fitmodeltrue_ents(T_vs_T̂_calcs_, temps_to_sweep)
    
    calc_all_fitmodel_negloglikesT̂( T_vs_T̂_calcs_, temps_to_sweep)
    
    calc_all_dkl_p_rand_p_true( T_vs_T̂_calcs_, temps_to_sweep )
    
    return T_vs_T̂_, T_vs_T̂_calcs_
end
function get_nsamples_to_sweep_all( mu_vals_to_sweep, true_temps_to_sweep, sweep_type, m_by_n, nsamples_to_sweep )
    m,n = m_by_n
    nsamples_to_sweep_all = []
    nsamples_to_sweep_per_mu = ntuple( i->similar(true_temps_to_sweep) , length(mu_vals_to_sweep) )
    if sweep_type == "constant M"
        nsamples_to_sweep_all = reduce(vcat,  x*ones(length(true_temps_to_sweep)) for x in nsamples_to_sweep )
    end
    if sweep_type == "constant mu"
        _,Jnn_true = init_2dNN(m,n)
        ml_ex_bf = ExactMLBuffer(Jnn_true)
        ents_exact = similar( true_temps_to_sweep )
        for (ll,ttt) in enumerate(true_temps_to_sweep)
            ents_exact[ll] = entropy_exact(ml_ex_bf, Jnn_true, ttt)
        end
        # @show ents_exact

        # nsamples_to_sweep_per_mu = ntuple( i->similar(true_temps_to_sweep) , length(mu_vals_to_sweep) )
        for (km,μ) in enumerate(mu_vals_to_sweep)
            nsamples_to_sweep_per_mu[km] .= round.(exp.(μ.+ents_exact))
            append!(nsamples_to_sweep_all, nsamples_to_sweep_per_mu[km])
        end
    end
    return nsamples_to_sweep_per_mu, nsamples_to_sweep_all
end
function run_full_sweeps_exact( optionss::TauSweepOptions, savedir; keep_buffer=false  )

    (; nreps, mu_vals_to_sweep, true_temps_to_sweep, sweep_type, m_by_n, nsamples_to_sweep, tau_to_sweep, nsamples_to_sweep_per_mu, nsamples_to_sweep_all ) = optionss
    m,n = m_by_n

    mcmc_or_exact_Tsamples="exact"
    mcmc_or_exact_T̂_samples="exact"
    fitmethod="exact"

    fit_opts = FitOptions2spinExact(Float64[]; relTol=10^(-5.), 
                    max_iter=15_000, showevery=1000)
    
    function run_sims_1M_1T( nreps, M_samples, T_true, μ, keep_buffer=keep_buffer ) 
        nsampless = M_samples*ones(nreps)
        @show nsampless
        for (kkj,nsampss) in enumerate(nsampless)
            if isfile( joinpath(savedir, "$(m)by$(n)_mu=$(μ)_T=$(T_true)_nsamps=$(nsampss)_$kkj.jld2") )
                println( joinpath(savedir, "$(m)by$(n)_mu=$(μ)_T=$(T_true)_nsamps=$(nsampss)_$kkj.jld2")*" already done... skipping")
                continue
            end
            println()
            println("rep = $kkj")
            println()
            time_elapsed=Int(round((time()-t1)/60))
            println("TIME ELAPSED = $time_elapsed min")
            
            println("\n RUNNING SWEEPS FOR N TRAINING SAMPLES = $nsampss")
            sampler_opt_Tsamples = SamplerOption(; 
                M=nsampss, 
                traceevery=100,
                showevery=2000,
                steps=100_000,
                seed=0,
                method=:MetropolisHastings,
                gpu= CUDA.functional(),
                FloatType = Float32, 
                stopafter=30
                )
            sampler_opt_forT̂_samples = SamplerOption(; 
                M=10, 
                traceevery=100,
                showevery=2000,
                steps=100_000,
                seed=0,
                method=:MetropolisHastings,
                gpu = CUDA.functional(),
                FloatType = Float32, 
                stopafter=30
                )


            _, T_vs_T̂_calcs_ = full_sweeps_and_fit( (m,n) , T_true , nsampss,
                                    sampler_opt_Tsamples,
                                    sampler_opt_forT̂_samples; 
                                    mcmc_or_exact_Tsamples=mcmc_or_exact_Tsamples,
                                    mcmc_or_exact_T̂_samples=mcmc_or_exact_T̂_samples,
                                    fitmethod=fitmethod, 
                                    fit_options=fit_opts,
                                    T̂_to_sweep=tau_to_sweep)
            
            if keep_buffer == false
                T_vs_T̂_calcs_["ml_exact_buffer"]= 0.0
            end
            
            T_vs_T̂_calcs_["mu"] = μ

            JLD2.save(joinpath(savedir, "$(m)by$(n)_mu=$(μ)_T=$(T_true)_nsamps=$(nsampss)_$kkj.jld2"), T_vs_T̂_calcs_ )

            # append!(tttlll ,[T_vs_T̂_calcs_]) 
        end
    end
    
    t1 =time()
    # for later use when pulling sweeps
    
    if sweep_type == "constant M"
        for T_true in true_temps_to_sweep
            for M in nsamples_to_sweep
                run_sims_1M_1T(nreps, M, T_true, "NA" )
            end
        end
    end   
    
    if sweep_type == "constant mu"
        for (km,μ) in  enumerate(mu_vals_to_sweep)
            for (M_samps, T_true) in zip( nsamples_to_sweep_per_mu[km] , true_temps_to_sweep )
                run_sims_1M_1T(nreps, Int(round(M_samps)), T_true, μ )
            end
        end   
    end
end
function get_nsamples_to_sweep_all( mu_vals_to_sweep, true_temps_to_sweep, sweep_type, m_by_n, nsamples_to_sweep )
    m,n = m_by_n
    nsamples_to_sweep_all = []
    nsamples_to_sweep_per_mu = ntuple( i->similar(true_temps_to_sweep) , length(mu_vals_to_sweep) )
    if sweep_type == "constant M"
        nsamples_to_sweep_all = reduce(vcat,  x*ones(length(true_temps_to_sweep)) for x in nsamples_to_sweep )
        # nsamples_to_sweep_per_mu = ntuple( i->fill("NA",length(true_temps_to_sweep)) , length(nsamples_to_sweep) )
    end
    if sweep_type == "constant mu"
        _,Jnn_true = init_2dNN(m,n)
        ml_ex_bf = ExactMLBuffer(Jnn_true)
        ents_exact = similar( true_temps_to_sweep )
        for (ll,ttt) in enumerate(true_temps_to_sweep)
            ents_exact[ll] = entropy_exact(ml_ex_bf, Jnn_true, ttt)
        end
        # @show ents_exact

        for (km,μ) in enumerate(mu_vals_to_sweep)
            nsamples_to_sweep_per_mu[km] .= round.(exp.(μ.+ents_exact))
            append!(nsamples_to_sweep_all, nsamples_to_sweep_per_mu[km])
        end
    end
    return nsamples_to_sweep_per_mu, nsamples_to_sweep_all
end
function pull_traces( T_vs_T̂_calcs, Ttrue)
    tau_swept = T_vs_T̂_calcs["T=$Ttrue"][:fit_model][:T̂_to_sweep]
    cross_ent_blue = T_vs_T̂_calcs["T=$Ttrue"][:fit_model][:negloglikes_under_gt]
    entropy_blue = T_vs_T̂_calcs["T=$Ttrue"][:fit_model][:fit_models_true_ent]
    dkl_blue = cross_ent_blue .- entropy_blue
    
    entropy_red  = T_vs_T̂_calcs["T=$Ttrue"][:true_ent]
    cross_ent_red = T_vs_T̂_calcs["T=$Ttrue"][:fit_model][:fitmodels_negloglikesT̂]
    dkl_red = cross_ent_red .- entropy_red
    
    dkl_prand_ptrue=T_vs_T̂_calcs["T=$Ttrue"][:dkl_pr_pt]
    N=T_vs_T̂_calcs["θ"].N
    dkl_ptrue_prand=-entropy_red+N*log(2)
    
    Jfit = T_vs_T̂_calcs["T=$Ttrue"][:fit_model][:Ĵ]
    
    return  cross_ent_blue, entropy_blue, dkl_blue, cross_ent_red, dkl_red, Jfit, tau_swept, entropy_red, dkl_prand_ptrue, dkl_ptrue_prand
end
function make_containers_all_reps( lbls, nreps, tau_swept, nspins )
    all_reps = Dict()
    for (k,lb) in enumerate(lbls[1:6])
        all_reps[ lb ] = zeros( nreps, length(tau_swept) )
        if k == 6
            all_reps[ lb ] = zeros( nreps, nspins, nspins )
        end
    end
    return all_reps
end
function append_all_reps!( all_reps, info_from_sweep, lbls, which_rep) 
    for (k,lb) in enumerate(lbls[1:5])
        all_reps[ lb ][which_rep,:] .= info_from_sweep[k]
    end
    all_reps[ lbls[6] ][which_rep,:,:] .= info_from_sweep[6]
end
function append_all_reps_rest( all_reps, info_from_sweep, lbls )
    for (k,lb) in enumerate(lbls[7:end])
        all_reps[ lb ] = info_from_sweep[k+6]
    end
end
function vectorize_all_reps( all_reps , lbls )
    for (k,lb) in enumerate(lbls[1:7])
        all_reps[ lb ] = [all_reps[ lb ]]
    end
end
function import_sweep_dicts( dir_name, M, T, mu )
    NSAMPS=Float64(M)
    all_sweep_files_strings=readdir(dir_name)
    tttlll = []
    for swp_file_name in all_sweep_files_strings
        swp_file_T = parse(Float64, split(swp_file_name, "_")[3][3:end])
        if occursin(string(NSAMPS) , swp_file_name ) && ((swp_file_T==T) && occursin( string(mu), swp_file_name))
            T_vs_T̂_calcs_=JLD2.load(joinpath(dir_name,swp_file_name ) )
            append!(tttlll ,[T_vs_T̂_calcs_]) 
            # @show swp_file_name
            # mu = parse(Float64,split( swp_file_name, "_" )[2][4:end])
        end
    end
    # @show size( tttlll)
    return tttlll
end
# for a given T and M
    # pull the following:
    # make a function that
    # gets all reps for cross_ent_red, cross_ent_blue, 
    # entropy_blue,
    # entropy_red (just a scalar), dkl_blue, 
    # dkl_red, t-hat_swept
    # put into a dataframe that keeps all raw data

function load_from_sweeps( tau_sweep_opts::TauSweepOptions, savedir )
    
    lbls = [:cross_ent_blue, :entropy_blue, :dkl_blue, 
        :cross_ent_red, :dkl_red, :J_fit,
        :tau_swept, :entropy_red, :dkl_prand_ptrue, :dkl_ptrue_prand ]

    n_mus=length(tau_sweep_opts.mu_vals_to_sweep)
    true_temps_to_sweep = tau_sweep_opts.true_temps_to_sweep
    nreps = tau_sweep_opts.nreps
    tau_swept = tau_sweep_opts.tau_to_sweep
    nspins = reduce(*,tau_sweep_opts.m_by_n)
    all_mus_swept=  reduce(vcat, 
                        fill(x, length(true_temps_to_sweep)) 
                        for x in tau_sweep_opts.mu_vals_to_sweep);
    nsamples_to_sweep_all = tau_sweep_opts.nsamples_to_sweep_all

    Df_raw_all_reps = DataFrame([name => [] for name in vcat([:M,:T,:mu],lbls) ] )

    for (M,Ttrue,mu) in zip(nsamples_to_sweep_all, repeat(true_temps_to_sweep, n_mus), all_mus_swept) 
        sweep_dicts_one_M = import_sweep_dicts( savedir, M,Ttrue, mu )
            # for 1 value of T and 1 value of M, pull all replicates of sims
        all_reps = make_containers_all_reps( lbls, nreps, tau_swept, nspins )
        all_reps[:M] = M
        all_reps[:T] = Ttrue
        all_reps[:mu] = mu
        # @show all_reps
        # @show size(sweep_dicts_one_M)
        # @show sweep_dicts_one_M
        for (k, T_and_tau_dict) in enumerate( sweep_dicts_one_M )
            info_to_pull = pull_traces( T_and_tau_dict, Ttrue )
            append_all_reps!( all_reps, info_to_pull, lbls, k )
            # @show k
            if k == nreps
                append_all_reps_rest( all_reps, info_to_pull, lbls )
            end
        end
        vectorize_all_reps( all_reps , lbls )
        # put dictionary with all replicates into dataframe

        df_1row = DataFrame( all_reps )
        append!(Df_raw_all_reps, df_1row )
    end 
    return Df_raw_all_reps
end
function get_splines(x, y; xfinestep=0.01)
    spl = Spline1D(x, y, bc="zero",s=.0,k=4)
    xfine = collect(minimum(x):xfinestep:maximum(x))
    ynew = evaluate(spl,xfine);
    return [xfine], [ynew]
end
function get_splines_1set_of_reps( reps, tau_swept )
    tfine, _ = get_splines(tau_swept, reps[1,:] )
    # @show tau_swept
    nreps=size(reps,1)
    all_splines = zeros(nreps,length(tfine[]))
    for (i,rr) in zip(1:nreps,eachrow(reps))
        tfine, all_spl = get_splines(tau_swept, rr)
        all_splines[i,:] = all_spl[]
    end
    return [tfine], [all_splines]
end
function get_means_and_stds( all_reps )
    return [mean( all_reps, dims=1 )], [std( all_reps, dims=1 )]
end
function get_means_and_splines_df( tau_sweep_opts::TauSweepOptions, df_raw_all_reps )

    df_means_and_splines=DataFrame()

    lbls = [:cross_ent_blue, :entropy_blue, :dkl_blue, 
            :cross_ent_red, :dkl_red, :J_fit,
            :tau_swept, :entropy_red, :dkl_prand_ptrue, :dkl_ptrue_prand ]
    
    all_rep_labels = lbls[1:5]

    n_mus=length(tau_sweep_opts.mu_vals_to_sweep)
    true_temps_to_sweep = tau_sweep_opts.true_temps_to_sweep
    nreps = tau_sweep_opts.nreps
    tau_swept = tau_sweep_opts.tau_to_sweep
    nspins = reduce(*,tau_sweep_opts.m_by_n)
    all_mus_swept=reduce(vcat, fill(x,length(true_temps_to_sweep)) 
        for x in tau_sweep_opts.mu_vals_to_sweep);
    nsamples_to_sweep_all = tau_sweep_opts.nsamples_to_sweep_all


    for (M,T, mu) in zip(nsamples_to_sweep_all, repeat(true_temps_to_sweep, n_mus), all_mus_swept) 
        # filter df for proper row
        temp_filt(m,t,μ) = all( (m,t,μ).== (M,T,mu) )
        df_filt = filter( [:M, :T,:mu] => temp_filt, df_raw_all_reps )
        # @show size(df_filt)
        # init a temp dictionary for holding everything
        temp_dict = Dict()
        tfine_for_splines = []
        tau_swept = df_filt[!,:tau_swept ][1]
        for lb in all_rep_labels
            all_reps_temp = df_filt[!, lb][1]
            # @show size(all_reps_temp)
            # calc avg and std of reps
            mean_lbl = Symbol(string(lb)*"_mean")
            std_lbl = Symbol(string(lb)*"_std")
            mm, ss = get_means_and_stds( all_reps_temp )
            temp_dict[mean_lbl], temp_dict[std_lbl] = (mm, ss)
            # make splines of all reps
            spl_lbl = Symbol(string(lb)*"_splines")
            tfine_for_splines, alll_splines = get_splines_1set_of_reps( all_reps_temp, tau_swept )
            temp_dict[ spl_lbl ] = alll_splines
            # calc avg and std of splines of all reps
            spl_lbl_mean = Symbol(string(spl_lbl)*"_mean")
            spl_lbl_std =  Symbol(string(spl_lbl)*"_std")
            temp_dict[ spl_lbl_mean ], temp_dict[ spl_lbl_std ] = get_means_and_stds( temp_dict[ spl_lbl ][] )
        end
        temp_dict[:tfine_for_splines]=tfine_for_splines
        temp_dict[:M] = M
        temp_dict[:T] = T
        temp_dict[:mu]=mu
         # append to df
        append!(df_means_and_splines, DataFrame(temp_dict))
    end  
    
    return df_means_and_splines
end
function searchsortednearest(a,x)
   idx = searchsortedfirst(a,x)
   if (idx==1); return idx; end
   if (idx>length(a)); return length(a); end
   if (a[idx]==x); return idx; end
   if (abs(a[idx]-x) < abs(a[idx-1]-x))
      return idx
   else
      return idx-1
   end
end
function get_vals_from_splines( all_reps_splines, tfine_for_splines, 
        at_tau_equals=[], get_opt=true )
    # returns values of splines at given values and at opt (argmin) value if requested
    nreps, _ =size(all_reps_splines)
    taus = tfine_for_splines
    
    dkls_at_taus = ntuple( i -> zeros(nreps), length(at_tau_equals) )
    # @show dkls_at_taus
    
    if !isempty( at_tau_equals )
        for (hh, at_tau) in enumerate(at_tau_equals)
            # @show at_tau
            # @show taus
            at_tau = (at_tau in taus) ? at_tau : taus[searchsortednearest( taus, at_tau )]
            at_tau_idx= findfirst( taus .==  at_tau )
            # @show at_tau_idx
            # @show size(all_reps_splines)
            for ll in 1:nreps
                dkls_at_taus[hh][ll] = all_reps_splines[ll,at_tau_idx]
            end
        end
    end
    
    tau_opts_all_reps = zeros(nreps)
    y_opts_all_reps = zeros(nreps)
    if get_opt == true
        for (k,rr) in enumerate(eachrow(all_reps_splines))
            minvall, indexx = findmin(rr)
            tau_opts_all_reps[k] = taus[indexx]
            y_opts_all_reps[k] = minvall #for dkls
        end
    end
    
    if isempty(at_tau_equals)
        return tau_opts_all_reps, y_opts_all_reps
    end
    if get_opt == true && !isempty(at_tau_equals)
        return tau_opts_all_reps, y_opts_all_reps, dkls_at_taus
    end
    if get_opt == false && !isempty(at_tau_equals)
        return dkls_at_taus
    end
end
# make a dataframe for tau_opts and ΔDkls
function get_tau_opt_df( tau_sweep_opts::TauSweepOptions, df_means_and_splines )

    df_tau_opt_DeltaDkl = DataFrame()

    lbls = [:cross_ent_blue, :entropy_blue, :dkl_blue, 
                :cross_ent_red, :dkl_red, :J_fit,
                :tau_swept, :entropy_red, :dkl_prand_ptrue, :dkl_ptrue_prand ]

    dkl_labels = [:dkl_blue, :dkl_red ]
    
    n_mus=length(tau_sweep_opts.mu_vals_to_sweep)
    true_temps_to_sweep = tau_sweep_opts.true_temps_to_sweep
    nreps = tau_sweep_opts.nreps
    tau_swept = tau_sweep_opts.tau_to_sweep
    nspins = reduce(*,tau_sweep_opts.m_by_n)
    all_mus_swept=reduce(vcat, fill(x,length(true_temps_to_sweep)) 
        for x in tau_sweep_opts.mu_vals_to_sweep);
    nsamples_to_sweep_all = tau_sweep_opts.nsamples_to_sweep_all


    for (M,T,mu) in zip(nsamples_to_sweep_all, repeat(true_temps_to_sweep, n_mus), all_mus_swept) 
        # filter df with all splines
        temp_filt(m,t,μ) = all( (m,t,μ).== (M,T,mu) )
        df_filt = filter( [:M, :T, :mu] => temp_filt, df_means_and_splines )
        # init a temp dictionary for holding everything
        temp_dict = Dict()
        # @show df_filt[!,:tfine_for_splines]
        tfine_for_splines=df_filt[!,:tfine_for_splines][][]
        for dkl_lbl in dkl_labels
            all_splines_reps = df_filt[!,Symbol(string(dkl_lbl)*"_splines")][]
            tau_opts_all, dkl_opt_all, dkl_at_1 = get_vals_from_splines( all_splines_reps, 
                                                tfine_for_splines, [1.], true)
            temp_dict[Symbol(string(dkl_lbl)*"_tau_opts")]=[tau_opts_all]
            temp_dict[Symbol(string(dkl_lbl)*"_dkl_opts")]=[dkl_opt_all]
            temp_dict[Symbol(string(dkl_lbl)*"_dkl_at_tau1")]=[dkl_at_1[1]]

            kkeys=[Symbol(string(dkl_lbl)*"_tau_opts"), Symbol(string(dkl_lbl)*"_dkl_opts"), Symbol(string(dkl_lbl)*"_dkl_at_tau1")]
            for khk in kkeys
                temp_dict[Symbol(string(khk)*"_mean")]=mean(temp_dict[khk][])
                # @show temp_dict[khk]
                temp_dict[Symbol(string(khk)*"_std")]=std(temp_dict[khk][])
            end
            temp_dict[Symbol(string(dkl_lbl)*"Delta_Dkl_mean")] = mean( temp_dict[Symbol(string(dkl_lbl)*"_dkl_at_tau1")][] .- temp_dict[Symbol(string(dkl_lbl)*"_dkl_opts")][] )
            temp_dict[Symbol(string(dkl_lbl)*"Delta_Dkl_std") ] = std( temp_dict[Symbol(string(dkl_lbl)*"_dkl_at_tau1")][] .- temp_dict[Symbol(string(dkl_lbl)*"_dkl_opts")][] )
        end
        temp_dict[:M]=M
        temp_dict[:T]=T
        temp_dict[:mu]=mu
        append!(df_tau_opt_DeltaDkl, DataFrame(temp_dict) )
    end
    return df_tau_opt_DeltaDkl
end
function decompose_dkls_ising( J_gt, J_fit, ml_bf, e_bf, T_true )

    all_e_dict = get_all_e_dict( J_gt, J_fit, ml_bf, e_bf )

    avg_prob_per_e_level = Dict(:fit => [], :gt=>[])
    rel_surprise_per_e_level = Dict(:fit => [], :gt=>[])
    rel_surpise = Dict(:fit => [], :gt=>[])
    
    # @show all_e_dict[:e_levels]

    for (i,e) in enumerate(all_e_dict[:e_levels])
        for keyy in [:fit, :gt]
            avg_prob_i =  get_avg_prob_per_level( i, all_e_dict, ml_bf ,T_true; fit_or_gt = keyy)
            append!( avg_prob_per_e_level[keyy] , avg_prob_i )
            
            avg_rel_surpise_i = get_avg_rel_surprise( i, all_e_dict, ml_bf,T_true ; fit_or_gt = keyy)
            append!( rel_surprise_per_e_level[keyy] , avg_rel_surpise_i )
            
            append!( rel_surpise[keyy] , avg_rel_surpise_i*all_e_dict[:nstates_per_level][i] )
        end
    end

    return all_e_dict, avg_prob_per_e_level, rel_surprise_per_e_level, rel_surpise
end
function get_all_e_dict( J_gt, J_fit, ml_bf, e_bf )
    if size(J_gt) != size(J_fit)
        @error "J_gt and J_fit are not the same dimensions!"
    end
    (; all_states, all_states_energy, exp_neg_energies, Z) = ml_bf
    all_e_gt, all_e_fit = ntuple( i->copy(all_states_energy), 2)
    
    all_e_gt  .= energy2spin( all_states, J_gt, e_bf ) 
    all_e_fit .= energy2spin( all_states, J_fit, e_bf ) 
    g_e = countmap(all_e_gt)
    es = []
    gs = []
    for (k,v) in sort(g_e)
        append!(es, k)
        append!(gs, v)
    end
    all_e_dict = Dict( :J_fit => J_fit,
        :J_gt => J_gt,
        :all_e_gt => all_e_gt,
        :all_e_fit => all_e_fit,
        :e_levels => es,
        :nstates_per_level => gs,
        :n_levels => length(es)
        )
    return all_e_dict
end
function get_avg_prob_per_level( i, all_e_dict, ml_bf, T_true ; fit_or_gt = :fit) 
    # i indexes the ith energy level under the ground truth
    (; exp_neg_energies, Z) = ml_bf
    
    # define prob dist
    dict_key = Symbol( "all_e_"*string(fit_or_gt) )
    exp_neg_energies .= exp.( -all_e_dict[dict_key] )
    exp_neg_energies = fit_or_gt == :gt ? exp.(-all_e_dict[dict_key]./T_true) : exp_neg_energies
    Z.=sum(exp_neg_energies)
    
    S_i = vec(all_e_dict[:all_e_gt] .== all_e_dict[:e_levels][i])
    
    return sum( exp_neg_energies[S_i] ) / Z[] / all_e_dict[:nstates_per_level][i]
end
function get_avg_rel_surprise( i, all_e_dict, ml_bf, T_true; fit_or_gt = :fit) 
    (; exp_neg_energies, Z) = ml_bf
    
    exp_neg_energies .= exp.(-all_e_dict[:all_e_fit])
    Z.=sum(exp_neg_energies)
    qs = exp_neg_energies./ Z
    
    # indices of states with E = E_i
    S_i = vec(all_e_dict[:all_e_gt] .== all_e_dict[:e_levels][i])
    
    # true prob of E_i
    Z_gt = sum( all_e_dict[:nstates_per_level] .* exp.(-all_e_dict[:e_levels]./T_true) )
    p_i_gt = exp(-all_e_dict[:e_levels][i]/T_true) / Z_gt
    
    if fit_or_gt == :fit
        #return avg over fit model
        avg_rel_surpise = sum( qs[S_i] .* log.( qs[S_i] ./ p_i_gt ) )
        return avg_rel_surpise / all_e_dict[:nstates_per_level][i]
    elseif fit_or_gt == :gt
        avg_rel_surpise = sum( p_i_gt .* log.( p_i_gt ./ qs[S_i] ) )
        return avg_rel_surpise / all_e_dict[:nstates_per_level][i]
    end
end



            # take set of all splines for all reps
        # one function for the below
            # take tau_opt of each rep, put in df
            # take dkl_opt of each rep, put in df
            # take dkl_1 of each rep, put in df
        # calc tau_opt_avg and tau_opt_std
        # calc ⟨ dkl_1 - dkl_opt ⟩ with avg taken over reps
            # stds also
        
