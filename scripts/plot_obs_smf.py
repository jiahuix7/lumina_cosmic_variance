import numpy as np

# conversion factors from Madau and Dickinson 2014
factor_from_Salpeter_to_Chabrier = 0.63
factor_from_Kroupa_to_Chabrier   = 0.63/0.67

def plot_obs_smf(redshift, ax, dict_style={}):
    if redshift == 3:
        f = np.genfromtxt("./obdata/SMF/Santini2012_z3.dat", names=True)
        ax.errorbar(f['logMstar'] + np.log10(factor_from_Salpeter_to_Chabrier), f['logPhi'],
            yerr=( f['logPhi'] - f['lo'], f['hi'] - f['logPhi']),
            marker='o', ms=dict_style['ms'], linestyle='none', color=dict_style['color'], mfc=dict_style['color'], mew=2, mec=dict_style['color'], lw=2, capsize=4, capthick=2, zorder=-1), #label=r"$\rm Santini+12$" if redshift==3 else None)

    if redshift == 3:
        f = np.genfromtxt("./obdata/SMF/llbert2013.dat", names=True) # Chabrier IMF
        ax.errorbar(f['logMstar'], f['logPhi'],
            yerr=( f['logPhi'] - f['lo'], f['hi'] - f['logPhi']),
            marker='o', ms=dict_style['ms'], linestyle='none', color=dict_style['color'], mfc=dict_style['color'], mew=2, mec=dict_style['color'], lw=2, capsize=4, capthick=2, zorder=-1), #label=r"$\rm Ilbert+13$" if redshift==3 else None)

        f = np.genfromtxt("./obdata/SMF/Davidzon2017.dat", names=True) # Chabrier IMF
        ax.plot(f['logMstar'], f['logPhi'],
            marker='o', ms=dict_style['ms'], linestyle='none', color=dict_style['color'], mfc=dict_style['color'], mew=2, mec=dict_style['color'], lw=2,  zorder=-1), #label=r"$\rm Davidzon+17$" if redshift==3 else None)

    f = np.genfromtxt("./obdata/SMF/Song2016.dat", names=True) # Salpeter IMF
    sel = f['z'] == redshift
    ax.errorbar(f['logMstar'][sel] + np.log10(factor_from_Salpeter_to_Chabrier), f['logPhi'][sel],
             yerr=(f['loerr'][sel], f['hierr'][sel]), marker='o', ms=dict_style['ms'],
             linestyle='none', color=dict_style['color'], mfc=dict_style['color'], mew=2, mec=dict_style['color'], lw=2, capsize=4, capthick=2,  zorder=-1),#label=r"$\rm Song+16$" #if redshift==3 else None)

    f = np.genfromtxt("./obdata/SMF/Stefanon2021.dat", names=True) # Salpeter IMF
    sel = f['z'] == redshift
    ax.errorbar(f['logMstar'][sel] + np.log10(factor_from_Salpeter_to_Chabrier), np.log10(f['Phi']*1e-4)[sel],
            yerr=( np.log10(f['Phi'])[sel] - np.log10(f['Phi']-f['loerr'])[sel], np.log10(f['Phi']+f['hierr'])[sel]-np.log10(f['Phi'])[sel]),
            marker='o', ms=dict_style['ms'], linestyle='none', color=dict_style['color'], mfc=dict_style['color'], mew=2, mec=dict_style['color'], lw=2, capsize=4, capthick=2, zorder=-1),# label=r"$\rm Stefanon+21$" #if redshift==3 else None)

    f = np.genfromtxt("./obdata/SMF/Weaver2023.dat", names=True) # Chabrier IMF
    sel = f['z'] == redshift
    ax.errorbar(f['logMstar'][sel], f['logPhi'][sel],
            yerr=( f['logPhi'][sel] - f['lo'][sel], f['hi'][sel] - f['logPhi'][sel]),
            marker='o', ms=dict_style['ms'], linestyle='none', color=dict_style['color'], mfc=dict_style['color'], mew=2, mec=dict_style['color'], lw=2, capsize=4, capthick=2, zorder=-1),#label=r"$\rm Weaver+23$" #if redshift==3 else None)

    f = np.genfromtxt("./obdata/SMF/Shuntov2025.dat", names=True)
    sel = f['z'] == redshift
    ax.errorbar(f['logMstar'][sel], f['logPhi'][sel],
            yerr=( f['logPhi'][sel] - f['lo'][sel], f['hi'][sel] - f['logPhi'][sel]),
            marker='o', ms=dict_style['ms'], linestyle='none', color=dict_style['color'], mfc=dict_style['color'], mew=2, mec=dict_style['color'], lw=2, capsize=4, capthick=2, zorder=-1),# label=r"$\rm Shuntov+25$" #if redshift==3 else None)

    ## JWST
    f = np.genfromtxt("./obdata/SMF/Wiebel2024.dat", names=True) # Kroupa IMF
    sel = (f['zmed'] == redshift) & (f['logPhi_loerr']<1e10)
    ax.errorbar(f['logMstar'][sel] + np.log10(factor_from_Kroupa_to_Chabrier), f['logPhi'][sel],
             yerr=(f['logPhi_loerr'][sel], f['logPhi_uperr'][sel]), marker='o', ms=dict_style['ms'],
             linestyle='none', color=dict_style['color'], mfc=dict_style['color'], mew=2, mec=dict_style['color'], lw=2, capsize=4, capthick=2,  zorder=3), #label=r"$\rm Weibel+24$" #if redshift==3 else None)
    sel = (f['zmed'] == redshift) & (f['logPhi_loerr']>1e10)
    ax.errorbar(f['logMstar'][sel] + np.log10(factor_from_Kroupa_to_Chabrier), f['logPhi'][sel],
             yerr=(0.3*np.ones(len(f['logMstar'][sel])), np.zeros(len(f['logMstar'][sel]))), marker='o', ms=dict_style['ms'],
             linestyle='none', color=dict_style['color'], mfc=dict_style['color'], mew=2, mec=dict_style['color'], lw=2, capsize=4, capthick=1,  zorder=3, uplims=True)

    f = np.genfromtxt("./obdata/SMF/Harvey2025.dat", names=True)
    sel = (f['zmin'] == redshift - 0.5) & (f['Phi']-f['Phi_loerr']>0)
    ax.errorbar(f['logMstar'][sel] + np.log10(factor_from_Kroupa_to_Chabrier), np.log10(f['Phi']*1e-4)[sel],
            yerr=( np.log10(f['Phi'][sel]) - np.log10(f['Phi']-f['Phi_loerr'])[sel], np.log10(f['Phi']+f['Phi_uperr'])[sel] - np.log10(f['Phi'][sel])),
            marker='o', ms=dict_style['ms'], linestyle='none', color=dict_style['color'], mfc=dict_style['color'], mew=2, mec=dict_style['color'], lw=2, capsize=4, capthick=2, zorder=3, label=r"$\rm Harvey+25$") #if redshift==3 else None)
    sel = (f['zmin'] == redshift - 0.5) & (f['Phi']-f['Phi_loerr']==0)
    ax.errorbar(f['logMstar'][sel] + np.log10(factor_from_Kroupa_to_Chabrier), np.log10(f['Phi']*1e-4)[sel],
            yerr=( 0.3 * np.ones(len(f['logMstar'][sel])), np.log10(f['Phi']+f['Phi_uperr'])[sel] - np.log10(f['Phi'][sel])),
            marker='o', ms=dict_style['ms'], linestyle='none', color=dict_style['color'], mfc=dict_style['color'], mew=2, mec=dict_style['color'], lw=2, capsize=4, capthick=1, zorder=3, uplims=True)

    f = np.genfromtxt("./obdata/SMF/Wang2025.dat", names=True)
    sel = f['z'] == redshift
    ax.errorbar(f['logMstar'][sel] + np.log10(factor_from_Kroupa_to_Chabrier), f['logPhi'][sel],
            yerr=( f['logPhi'][sel] - f['lo'][sel], f['hi'][sel] - f['logPhi'][sel]),
            marker='o', ms=dict_style['ms'], linestyle='none', color=dict_style['color'], mfc=dict_style['color'], mew=2, mec=dict_style['color'], lw=2, capsize=4, capthick=2, zorder=3), #label=r"$\rm Wang+25$" #if redshift==3 else None)
