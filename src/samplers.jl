function entropy(t::AbstractArray{<:Real,N}) where N
    τ = sum(t)
    τ ≈ one(τ) || @warn "Probabilities sum to $τ. Results may not make sense!"
    return Flux.Losses.crossentropy(t, t; dims = 1:N)
end

function _mutualinfo(t::AbstractMatrix)
    t_x, t_y = sum(t, dims = 2), sum(t, dims = 1)
    return entropy(t_x) + entropy(t_y) - entropy(t)
end

function pair_information(X_μim::T, Y_μim::T) where T<:AbstractArray{<:Real,3}
    q, N, M = size(X_μim)
    X, Y = reshape(X_μim, :,M), reshape(Y_μim, :,M)
    t = 1//M * (X * transpose(Y))
    t = reshape(t, q,N,q,N)
    return maximum(_mutualinfo(@view(t[:,i,:,i])) for i ∈ 1:N)
end



@with_kw struct SamplerOption
    M::Int = 2000
    traceevery::Int = 500
    showevery::Int = 2000
    steps::Int = 100_000
    seed::Int = 123
    method::Symbol = :MetropolisHastings
    gpu::Bool = CUDA.functional()
    FloatType::DataType = Float32
    stopafter::Int = 40 #stop sampling after (stopafter)*(traceevery) steps in which Emean has not decreased
end
function _type2dict(x::SamplerOption)
    di = type2dict(x)
    delete!(di, :showevery)
    return di
end

init_MetropolisHastings(zz, θ, temp) = init_mh(zz, θ, temp)
function init_mh(zz, θ, temp)
    # if typeof(θ)<:Sector
    #     energybuffer = SectorEnergyBuffer(z, θ)
    #     E = x -> rmul!(sectorenergy(x, θ, energybuffer), 1/temp)
    #     mh = MetropolisHastings(z)
    #     return  mh, E
    # else
        energybuffer = EnergyBuffer(zz, θ)
        E = x -> rmul!(energy(x, θ, energybuffer), 1/temp)
        mh = MetropolisHastings(zz)
        return  mh, E
    # end
end

init_GibbsWithGradients(zz, θ, temp) = init_gwg(zz, θ, temp)
function init_gwg(zz, θ, temp)
    energybuffer = EnergyBuffer(zz, θ)
    ∂E = x -> (
        E  = energy(x, θ, energybuffer); 
        ∇E = ∂x_energy(x, θ, energybuffer); 
        ( rmul!(E, 1/temp) , rmul!(∇E, 1/temp) )
    )
    gwg = GibbsWithGradients(zz)
    return  gwg, ∂E
end


function drawsamples(θ, temp=1.0, args::SamplerOption = SamplerOption() )
    (; M, traceevery, showevery, steps, seed, method, gpu, FloatType, stopafter) = args
    iszero(showevery%traceevery) || @error "showevery MUST be a multiple of traceevery"
    
    @printf "----------------------------------------\n"
    iszero(seed) || Random.seed!(seed)
    
    (; q,N) = θ
    
    z₀ = FloatType.(Flux.onehotbatch(rand(1:q, N, M), 1:q))
    z₀ = args.gpu ? cu(z₀) : z₀
    @printf "%s\n" "Array type: $(typeof(z₀))"
    z = copy(z₀)
    
    temp = FloatType.(temp)
    sampler, E = eval( Symbol("init_"*String(method)) )(z, θ, temp)
    
    @printf "------ Begin Metropolis Hasting ------\n"
    #initialize traces
    t0, I_x_x0, E_mean, E_std, accrate = time(), FloatType[], FloatType[], FloatType[], FloatType[]
    
    flips = 0.0
    E_mean_min, counter = FloatType(Inf), FloatType(0)
    
    for i ∈ 0:steps
        i > 0 && sampler(z, E)
        flips += iszero(i) ? 0 : mean(acceptrate(sampler) )
        iszero(i) && @printf "   step  I(x(0);x(t))  mean(E)     std(E)    rate  <flips>     time\n"
        iszero(i) && @printf "-------  ------------  ----------  --------- ----  -------  -------\n"
        
        if iszero(i%traceevery)
            push!(I_x_x0, pair_information(z, z₀))
            push!(accrate, iszero(i) ? NaN : mean(acceptrate(sampler)))
            
            ith_energy = temp*E(z)
            ith_energy = eltype(ith_energy)<:AbstractArray ? ith_energy[1] : ith_energy
            E_mean_i = mean(ith_energy)
            push!(E_mean,  E_mean_i)
            push!( E_std, std(ith_energy) )
            # check if E_mean has decreased and stop if it hasnt for substantial amount of time
            if (E_mean[end] < E_mean_min) && (E_mean[end] ≉ E_mean_min)
                E_mean_min, earlystop = E_mean[end], i
                counter = 0
            else
                counter+= 1
                if counter > stopafter
                    @printf "E_mean hasn't decreased for %i steps. Stop sampling!\n"  Int(counter*traceevery - traceevery)
                    break
            end
        end 
            
        end
        
        if iszero(i%showevery)
            @printf(
                "%7d  %12.7f  %10.3e  %9.3e  %4.2f  %7.1e  %4.0f\n", 
                i, I_x_x0[end], E_mean[end], E_std[end], accrate[end], flips[], time()-t0
            )
        end
        
        i == steps && @printf "------- Max Steps Reached. Ending Metropolis Hasting -------\n\n"
    end
    
    traces = Dict{Symbol,Any}(zip(
        (:I_x_x0, :E_mean, :E_std, :acceptrate), 
        (I_x_x0, E_mean, E_std, accrate)
    ))
    
    z = collect(z) .> 1//2
    return merge(traces, _type2dict(args) ), z
    
end

# only use below func for energy descent
function drawsamples!(z, θ, temp=1.0, args::SamplerOption = SamplerOption() )
    (; M, traceevery, showevery, steps, seed, method, gpu, FloatType, stopafter) = args
    iszero(showevery%traceevery) || @error "showevery MUST be a multiple of traceevery"
    
    method != :EnergyDescent && @error "Only sampling method valid for drawsamples! is energy descent"
    
    @printf "----------------------------------------\n"
    iszero(seed) || Random.seed!(seed)
    
    (; q,N) = θ
    
#     z₀ = FloatType.(Flux.onehotbatch(rand(1:q, N, M), 1:q))
    z = args.gpu ? cu(z) : z
    @printf "%s\n" "Array type: $(typeof(z))"
    z₀ = copy(z)
    
    sampler, ΔE, E = eval( Symbol("init_"*String(method)) )(z, θ, FloatType.(temp))
    
    @printf "------ Begin %s ------\n" String(method)
    #initialize traces
    t0, I_x_x0, E_mean, E_std, accrate = time(), FloatType[], FloatType[], FloatType[], FloatType[]
    
    flips = 0.0
    E_mean_min, counter = FloatType(Inf), FloatType(0)
    
    for i ∈ 0:steps
        i > 0 && (z.=sampler(z, ΔE))
        flips += iszero(i) ? 0 : mean(acceptrate(sampler) )
        iszero(i) && @printf "   step  I(x(0);x(t))  mean(E)     std(E)    rate  <flips>     time\n"
        iszero(i) && @printf "-------  ------------  ----------  --------- ----  -------  -------\n"
        
        if iszero(i%traceevery)
            push!(I_x_x0, pair_information(z, z₀))
            push!(accrate, iszero(i) ? NaN : mean(acceptrate(sampler)))
            
            ith_energy = E(z)
            ith_energy = eltype(ith_energy)<:AbstractArray ? ith_energy[1] : ith_energy
            E_mean_i = mean(ith_energy)
            push!(E_mean,  E_mean_i)
            push!( E_std, std(ith_energy) )
            # check if E_mean has decreased and stop if it hasnt for substantial amount of time
        end 
                    
        if iszero(i%showevery)
            @printf(
                "%7d  %12.7f  %10.3e  %9.3e  %4.2f  %7.1e  %4.0f\n", 
                i, I_x_x0[end], E_mean[end], E_std[end], accrate[end], flips[], time()-t0
            )
        end
    
        if accrate[end] == 0
                @printf "No more bits being flipped. End sampling"
                break
        end
        
        i == steps && @printf "------- Max Steps Reached. Ending Energy Descent -------\n\n"
    end
    
    traces = Dict{Symbol,Any}(zip(
        (:I_x_x0, :E_mean, :E_std, :acceptrate), 
        (I_x_x0, E_mean, E_std, accrate)
    ))
    
    z = collect(z) .> 1//2
    return merge(traces, _type2dict(args) ) 
    
end

# use below method to sample a model and track mutations at each time step
function drawsamples!!(z, θ, temp=1.0, args::SamplerOption = SamplerOption(); trace_samples = false )
    (; M, traceevery, showevery, steps, seed, method, gpu, FloatType, stopafter) = args
    iszero(showevery%traceevery) || @error "showevery MUST be a multiple of traceevery"
    
    method == :EnergyDescent && @error "Only sampling method valid for drawsamples!! is metropolis hastings and its variants"
    
    @printf "----------------------------------------\n"
    iszero(seed) || Random.seed!(seed)
    
    (; q,N) = θ
    
    local z₀
    z_c =[]
    z = args.gpu ? cu(z) : z
    z₀ = similar(z)
    z₀ .= z
    @printf "%s\n" "samples' array type: $(typeof(z₀))"
        
    sampler, E = eval( Symbol("init_"*String(method)) )(z₀, θ, FloatType.(temp))
    
    @printf "------ Begin %s ------\n" String(method)
    #initialize traces
    t0, I_x_x0, E_mean, E_std, accrate = time(), FloatType[], FloatType[], FloatType[], FloatType[]
    z_trace = []
    # z_trace = similar(z, q,N,steps+1)
    push!(z_trace, z)
    
    flips = 0.0
    E_mean_min, counter = FloatType(Inf), FloatType(0)
    
    @show z₀ == z
    for i ∈ 0:steps
        i > 0 && ( (sampler(z₀, E)) ; ( (  trace_samples == true && 
                    ( mean(acceptrate(sampler)) == 1 && 
                        (push!(z_trace, Float32.(z₀))) )  )) )
        # @show size(z₀)
        flips += iszero(i) ? 0 : mean(acceptrate(sampler) )
        iszero(i) && @printf "   step  I(x(0);x(t))  mean(E)     std(E)    rate  <flips>     time\n"
        iszero(i) && @printf "-------  ------------  ----------  --------- ----  -------  -------\n"
        
        
        push!(I_x_x0, 0f0)
        push!(accrate, iszero(i) ? NaN : mean(acceptrate(sampler)))
        # push!(z_trace, z_c)
        # i > 0 && @show z_trace[end-1] == z_trace[end]
        # if accrate[end] == FloatType(1)
        #     @show all(z_trace[end-1] == z_trace[end])
        # end
        ith_energy = E(z₀)
        ith_energy = eltype(ith_energy) <: AbstractArray ? ith_energy[1] : ith_energy
        E_mean_i = mean(ith_energy)
        push!(E_mean,  E_mean_i)
        push!( E_std, std(ith_energy) )
         
                    
        if iszero(i%showevery)
            @printf(
                "%7d  %12.7f  %10.3e  %9.3e  %4.2f  %7.1e  %4.0f\n", 
                i, I_x_x0[end], E_mean[end], E_std[end], accrate[end], flips[], time()-t0
            )
        end
        
        i == steps && @printf "------- Max Steps Reached. -------\n\n"
    end
    
    traces = Dict{Symbol,Any}(zip(
        (:I_x_x0, :E_mean, :E_std, :acceptrate, :z_trace), 
        (I_x_x0, E_mean, E_std, accrate, z_trace)
    ))
    
    z₀ = collect(z₀) .> 1//2
    return z₀, merge(traces, _type2dict(args) ) 
    
end


