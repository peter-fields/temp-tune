function plot_one_ising_replicate(ax, df_raw_all_reps, M, T, r, J_gt, ml_bf, e_bf; 
        leg3_loc=(0.16,0.535), e_lvls_to_show=(1,5))
    sss="indianred"
    coral_rgba=(matplotlib.colors.to_rgba(sss)[1:3]..., 0.5)
    coral_rgb1=matplotlib.colors.to_rgba(sss)
    steelblue_rgba=(matplotlib.colors.to_rgba("steelblue")[1:3]...,0.5)
    steelblue_rgba1=(matplotlib.colors.to_rgba("steelblue")[1:3]...,1)

    temp_filt(t,m) = all( (t,m).== (T,M) )
    df_filtered = filter( [:T, :M] => temp_filt, df_raw_all_reps );  

    J_fitt = df_filtered.J_fit[][r, :,:]

    τ_swept = df_filtered.tau_swept[]           

    all_e_dict, avg_prob_per_e_level, rel_surprise_per_e_level, rel_surpise =decompose_dkls_ising( 
                                                                    J_gt, J_fitt, ml_bf, e_bf, T );

    idcss=e_lvls_to_show[1]:e_lvls_to_show[2]
    xxss = collect(1:all_e_dict[:n_levels])
    # ax[1].bar( xxss , avg_prob_per_e_level[:gt].-avg_prob_per_e_level[:fit] )


    ggss = all_e_dict[:nstates_per_level][idcss]

    ax[1].bar( xxss[idcss] , all_e_dict[:nstates_per_level][idcss] , width=0.5, color="k" )
    ax[1].set_yscale("log")


    ax[2].bar(xxss[idcss].-0.135 , avg_prob_per_e_level[:gt][idcss], 
        width=0.25, color=(0,0,0,0.5),edgecolor="k", #hatch="......",
          linewidth=0.6,
        label=L"p_{T,i}")

    ax[2].bar(xxss[idcss].+0.135 , avg_prob_per_e_level[:fit][idcss] , 
        width=0.25, color=(0,0,0,0.1),edgecolor="k", #hatch="///////",
        linewidth=0.6,
     label=L"\hat{q}_{i}")

    ax[2].set_yscale("log")

    br=ax[3].bar( xxss[idcss].-0.19 , rel_surpise[:gt][idcss], width=0.25, 
            edgecolor=coral_rgb1, facecolor=coral_rgba,
                    label=L"D_{KL}(p_T || \hat q)\cap\Lambda_i", hatch="\\\\",
            rasterized = true)
    bb=ax[3].bar( xxss[idcss].+0.19 , rel_surpise[:fit][idcss] , width=0.25, 
        ec=steelblue_rgba1, facecolor=steelblue_rgba,
                    label=L"D_{KL}(\hat q||p_T)\cap\Lambda_i", hatch="\\\\" ,
        rasterized = true);
    for l in 1:length(bb)
        bb[l]._hatch_color, br[l]._hatch_color = ntuple(i->matplotlib.colors.to_rgba("white"),2)
    end

    ax[2].legend(frameon=false, 
        # fontsize=18, 
        framealpha=1,
        handlelength=1.33, handletextpad=0.2,
        labelspacing=0.2, loc=(0.64, 0.55))

    ax[3].set_xticks(xxss[idcss])
    ax[3].set_xticklabels(collect(idcss).-1)
    

    leg3 = ax[3].legend(frameon=false, 
        # fontsize=15, 
        loc=leg3_loc,
        # bbox_to_anchor=(0.475, 0.66), 
        handlelength=1.33, handletextpad=0.2,
        labelspacing=0.2)

    leg3_ = ax[3].get_legend()

    ax[1].set_ylabel("density\nof states")
        # labelpad=50)?

    ax[2].set_ylabel("average prob.\nmass per state")
    ax[3].set_xlabel("energy level index")
    
    ax[3].set_ylabel("nats", labelpad=6)
    
    for k in [1,2]
        ax[k].tick_params(labelbottom=false)
    end
    
end
function plot_tau_sweep_one_rep(ax, df_raw_all_reps, M, T, r, J_gt, ml_bf, e_bf;
                                    mss=5, lww=1, leg1_loc=(0.289,0.54),
                                    leg3_loc=(-0.02,0.44), exct_idx=3 )
    
    ax[2], ax[3] = (ax[3], ax[2])

    sss="indianred"
    coral_rgba=(matplotlib.colors.to_rgba(sss)[1:3]..., 0.5)
    coral_rgb1=matplotlib.colors.to_rgba(sss)
    steelblue_rgba=(matplotlib.colors.to_rgba("steelblue")[1:3]...,0.5)
    steelblue_rgba1=(matplotlib.colors.to_rgba("steelblue")[1:3]...,1)

    temp_filt(t,m) = all( (t,m).== (T,M) )
    df_filtered = filter( [:T, :M] => temp_filt, df_raw_all_reps );  

    J_fitt = df_filtered.J_fit[][r, :,:]
    
    τ_swept = df_filtered.tau_swept[]

    all_e_dict_tau1, _, _, _ = decompose_dkls_ising( 
                                                    J_gt, J_fitt, ml_bf, e_bf, T );

    suprise_per_level_of_tau = Dict( :gt=>zeros( all_e_dict_tau1[:n_levels], length(τ_swept) ),
                                    :fit=>zeros( all_e_dict_tau1[:n_levels], length(τ_swept) ))

    for (kk,ττ) in enumerate(τ_swept)
        all_e_dict, _, _, rel_surpise = decompose_dkls_ising( 
                                                         J_gt, J_fitt./ττ, ml_bf, e_bf, T );

        suprise_per_level_of_tau[:gt][:, kk] .= rel_surpise[:gt]
        suprise_per_level_of_tau[:fit][:, kk] .= rel_surpise[:fit]
    end 

    tau_star = τ_swept[findmin( vec(sum( suprise_per_level_of_tau[:fit][1:end,:],dims=1 ) ) )[2]]
    tau_prime = τ_swept[findmin( vec(sum( suprise_per_level_of_tau[:gt][1:end,:],dims=1 ) ) )[2]]


    ax[1].plot( τ_swept, vec(sum( suprise_per_level_of_tau[:gt][1:end,:],dims=1 ) ) ,
                color=coral_rgb1, label=L"D_{KL}(p_T || \hat q_\tau)",
                linewidth=lww)
    ax[1].plot( τ_swept, vec(sum( suprise_per_level_of_tau[:fit][1:end,:],dims=1 ) ),  
                color=steelblue_rgba1,label=L"D_{KL}(\hat q_\tau||p_T)"  ,
                linewidth=lww)
    # ax[1].grid("on", alpha=0.35)
    ax[1].vlines( 1, -1,5, color="k", alpha=0.05,
                linewidth=lww)
    ax[2].vlines( 1, -1,5, color="k", alpha=0.05,
                linewidth=lww)
    ax[3].vlines( 1, -1,5, color="k", alpha=0.05,
                linewidth=lww)
    ax[1].vlines( (tau_star,tau_prime), -1,5, color="k", alpha=0.15,
                linewidth=lww)
    ax[1].plot( [tau_star,tau_prime], [
            findmin( vec(sum( suprise_per_level_of_tau[:fit][1:end,:],dims=1) ) )[1]
            findmin( vec(sum( suprise_per_level_of_tau[:gt][1:end,:],dims=1 ) ) )[1] ],
            "k.", markersize=mss)

    ax[1].legend(loc=leg1_loc, 
        # fontsize=18, 
        framealpha=1, frameon=false, 
        # bbox_to_anchor=(1.025, -0.065),
        labelspacing=0.2, handletextpad=0.1, handlelength=1)

    ax[2].plot( τ_swept, vec(sum( suprise_per_level_of_tau[:gt][1:end,:],dims=1 ) ) ,
                color=coral_rgb1, alpha=0.5,
                linewidth=lww)
    
    
    ax[2].plot( τ_swept, vec(sum( suprise_per_level_of_tau[:gt][1:1,:],dims=1 ) ),
                linestyle="dotted", color="k", label="ground",
                linewidth=lww)
    
    ax[2].plot( τ_swept, vec(sum( suprise_per_level_of_tau[:gt][2:(exct_idx+1),:],dims=1 ) ),
                linestyle="dashed", color="k", label="excited 1 to $exct_idx",
                linewidth=lww)
    ax[2].plot( τ_swept, vec(sum( suprise_per_level_of_tau[:gt][(exct_idx+2):end,:],dims=1 ) ),
                linestyle="solid", color="k",label="excited $(exct_idx+1) and up",
                linewidth=lww)
    # ax[2].grid("on", alpha=0.2)
    ax[2].vlines( (tau_star,tau_prime), -1,5, color="k", alpha=0.15, linewidth=lww)
      ax[2].plot( [tau_star,tau_prime][2], [
            findmin( vec(sum( suprise_per_level_of_tau[:fit][1:end,:],dims=1) ) )[1]
            findmin( vec(sum( suprise_per_level_of_tau[:gt][1:end,:],dims=1 ) ) )[1] ][2],
            "k.", markersize=mss)
    # ax[2].legend(framealpha=1, loc="upper right", frameon=false)



    ax[3].plot( τ_swept, vec(sum( suprise_per_level_of_tau[:fit][1:end,:],dims=1 ) ) ,
                color=steelblue_rgba1, alpha=0.5,
                linewidth=lww)
    ax[3].plot( τ_swept, vec(sum( suprise_per_level_of_tau[:fit][1:1,:],dims=1 ) ),
                linestyle="dotted", color="k", label="ground",
                linewidth=lww)
    ax[3].plot( τ_swept, vec(sum( suprise_per_level_of_tau[:fit][2:(exct_idx+1),:],dims=1 ) ),
                linestyle="dashed", color="k", label="excited 1 to $exct_idx",
                linewidth=lww)
    ax[3].plot( τ_swept, vec(sum( suprise_per_level_of_tau[:fit][(exct_idx+2):end,:],dims=1 ) ),
                linestyle="solid", color="k", label="excited $(exct_idx+1) and up",
                linewidth=lww)
    ax[3].legend(framealpha=1, loc=leg3_loc, 
        frameon=false, handletextpad=0.2,
        # fontsize=15, 
        handlelength=1, labelspacing=0.1)
    # ax[3].legend()

    # ax[3].grid("on", alpha=0.35)
    ax[3].vlines( (tau_star,tau_prime), -1,5, color="k", alpha=0.15, linewidth=lww)
    ax[3].plot( [tau_star,tau_prime][1], [
            findmin( vec(sum( suprise_per_level_of_tau[:fit][1:end,:],dims=1) ) )[1]
            findmin( vec(sum( suprise_per_level_of_tau[:gt][1:end,:],dims=1 ) ) )[1] ][1],
            "k.", markersize=mss)
    
    for k in [1,3]
    ax[k].tick_params(labelbottom=false)
    end
    
    ax[1].set_ylabel("nats")
    # , fontsize=13)
    ax[2].set_ylabel(L"D_{KL}"*" per level")
        # , fontsize=13)
    ax[3].set_ylabel(L"D_{KL}"*" per level")

    ax[1].set_xlim(0.5,1.4)
    ax[1].set_ylim(-0.25,3.25)
    
end
function get_arrays_for_phase_plot(df_raw_all_reps, df_tau_opt_DeltaDkl)

    true_temps_to_swp = sort(unique(df_raw_all_reps.T))
    all_Ms_swpt = unique(df_raw_all_reps.M)

    all_taus=zeros( length(true_temps_to_swp)  , length(all_Ms_swpt) )
    all_taus_prime=zeros( length(true_temps_to_swp)  , length(all_Ms_swpt) )

    # tau opt blue plots
    for (j,t_true) in enumerate(true_temps_to_swp[1:1:end])
        df_temp=filter( :T => tt -> tt==t_true, df_tau_opt_DeltaDkl )

        # dkl_at_1 = df_temp.dkl_blue_dkl_at_tau1
        # @show( dkl_at_1[j] )
        tau_opts = df_temp.dkl_blue_tau_opts_mean
        tau_primes=df_temp.dkl_red_tau_opts_mean
        tau_opts_err = df_temp.dkl_blue_tau_opts_std

        all_taus[j,:].=tau_opts
        all_taus_prime[j,:].=tau_primes

        # dkl_opts=reduce( hcat, x for x in df_temp.dkl_blue_dkl_opts )
        # dkl_at_1 = reduce( hcat, x for x in df_temp.dkl_blue_dkl_at_tau1 )
        # @show size(dkl_opts)

        # nsamps_1T = df_temp.M
        # ground_true_entropies = filter( :T => tt -> tt==t_true, df_raw_all_reps ).entropy_red
        # M_over_exp_ent = nsamps_1T ./ (exp.(ground_true_entropies) ) 
        # @show (t_true, nsamps_1T)

        # df_temp_2=filter( :T => tt -> tt==t_true, df_raw_all_reps )
        # nsamps_1T = df_temp_2.M

        # ents_blue_at1_mean = zeros(length(nsamps_1T))
        # cross_ents_blue_at1_mean = zeros(length(nsamps_1T))
        # ents_blue_at1_std = zeros(length(nsamps_1T))
        # ents_blue_at1 = zeros(nreps, length(nsamps_1T))  

        # for (k, mm) in enumerate( nsamps_1T )
        #     tau_is1_idx=df_temp_2.tau_swept[k].==1
        #     ents_blue_at1[:,k] = df_temp_2.entropy_blue[k][:,tau_is1_idx]
        #     cross_ents_at1 = df_temp_2.cross_ent_blue[k][:,tau_is1_idx]
        #     cross_ents_blue_at1_mean[k] = mean(cross_ents_at1)
        #     ents_blue_at1_mean[k] = mean(ents_blue_at1[:,k])
        #     ents_blue_at1_std[k]  = std(ents_blue_at1[:,k])
        # end

        # df_temp_3=filter( :T => tt -> tt==t_true, df_means_and_splines )
        # ents_blue_at_tau_op = zeros(nreps, length(nsamps_1T))
        # for (k, mm) in enumerate( nsamps_1T )
        #     all_t_curr = df_temp_3.tfine_for_splines[k][]
        #     t_op_curr = df_temp.dkl_blue_tau_opts[k]
        #     ents_at_op = similar(t_op_curr)
        #     for j in 1:length(t_op_curr)
        #         t_op_id = all_t_curr .== t_op_curr[j]
        #         ent_curr = df_temp_3.entropy_blue_splines[k][j,t_op_id][]
        #         ents_at_op[j] = ent_curr
        #     end
        #     ents_blue_at_tau_op[:,k] .= ents_at_op
        # end

        # ents_blue_at_tau_op_mean = vec(mean(ents_blue_at_tau_op, dims=1))

        # mean_ent_reduction = vec(mean(ents_blue_at1.-ents_blue_at_tau_op, dims=1))
        # @show size(ents_blue_at_tau_op_mean)

        # M_over_exp_ent_from_fit = nsamps_1T ./ (exp.(ents_blue_at1_mean))

        # t_all=[2.,3]
        # cnow=get_c(cc, t_all, t_true)    
    end
    return true_temps_to_swp, all_Ms_swpt, all_taus, all_taus_prime
end
function tau_star_plots(ax,
                        true_temps_to_swp, all_Ms_swpt, all_taus, all_taus_prime;
                        Ms_idxx = 1:12, Ts_idxx = 1:28)
    epss=0.3
    
    

    mycmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "mycamppp",
        ["mediumpurple","w" , "darkseagreen"])
    normmm=PyPlot.matplotlib.colors.TwoSlopeNorm(vmin=minimum(all_taus[Ts_idxx,Ms_idxx]),
                                            vmax=3.,
                                            # vmax=maximum(all_taus[Ts_idxx,Ms_idxx]),
                                            vcenter=1)

    imm=ax.pcolor( all_Ms_swpt[Ms_idxx] , (true_temps_to_swp)[Ts_idxx],
                all_taus[Ts_idxx,Ms_idxx], cmap=mycmap, norm=normmm, rasterized=true)
            # vmin=1-epss,vmax=1+epss, clim=(0.6,3) )
            # extent=(1,5, 2,4.8))

    # cb=colorbar(imm, ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1., 2, 3])
    # cb.minorticks_on()

        # cb=colorbar(imm, orientation="horizontal", location="top",
        #     ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1., 2, 3])
        # cb.set_ticklabels([".5", ".6", ".7", ".8", ".9", "1", "2", "3"]) # custom text

    rr=ax.get_data_ratio()
    # ax.set_aspect(0.66*1/rr)
    ax.set_xscale("log")
    # ax.set_yscale("log")
    # all_taus[Ts_idxx,Ms_idxx]

    ax.set_ylabel("ground truth "*L"T")
    ax.set_xlabel("training sample size "*L"M")
    # cb.set_label(L"\tau^*", fontsize=13.5)

    ax.tick_params(axis="both")
    # ax.text("..", 10000,4)
         # ax.yaxis.set_minor_locator(AutoMinorLocator())
         # ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
    return imm
end
function tau_prime_plot(ax,
                        true_temps_to_swp, all_Ms_swpt, all_taus, all_taus_prime;
                        Ms_idxx = 1:12, Ts_idxx = 1:28)    

    mycmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "mycamppp",
        ["mediumpurple","w" , "darkseagreen"])
    normmm=PyPlot.matplotlib.colors.TwoSlopeNorm(vmin=minimum(all_taus[Ts_idxx,Ms_idxx]),
                                            vmax=3.,
                                            # vmax=maximum(all_taus[Ts_idxx,Ms_idxx]),
                                            vcenter=1)

# normmm=PyPlot.matplotlib.colors.TwoSlopeNorm(vmin=minimum(all_taus_prime[Ts_idxx,Ms_idxx]),
#                                         vmax=maximum(all_taus_prime[Ts_idxx,Ms_idxx]),
#                                         vcenter)

imm=ax.pcolor( all_Ms_swpt[Ms_idxx] , (true_temps_to_swp)[Ts_idxx],
            all_taus_prime[Ts_idxx,Ms_idxx], cmap=mycmap, norm=normmm, rasterized = true)
        # vmin=1-epss,vmax=1+epss, clim=(0.6,3) )
        # extent=(1,5, 2,4.8))

# cb.ax.set_xlabel(L"\tau'")
# cb.minorticks_on()

rr=ax.get_data_ratio()
# ax.set_aspect(0.66*1/rr)
ax.set_xscale("log")
# ax.set_yscale("log")
# all_taus[Ts_idxx,Ms_idxx]

ax.set_ylabel("ground truth "*L"T")
ax.set_xlabel("training data size "*L"M")


# ax.tick_params(axis="both",)

# cb.ax.tick_params()
     # ax.yaxis.set_minor_locator(AutoMinorLocator())
     # ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
    return imm
end