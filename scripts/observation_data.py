import numpy as np

def load_harikane2023_specz(redshift, type=0):
    """
    Load the data from Harikane et al. 2023 
    """
    f = np.genfromtxt('obdata/UVLF/Harikane2023_Specz.dat', names=True, delimiter=',')
    select = (f['z'] == redshift) & (f['type'] == type)
    log_density = np.log10(f['density'][select])
    density_lower = np.array([max(1e-10, x) for x in (f['density'][select] - f['lower_err'][select])])

    log_lower_err = log_density - np.log10(density_lower)
    log_upper_err = np.log10(f['density'][select] + f['upper_err'][select]) - log_density

    return f['ABmag'][select], log_density, log_lower_err, log_upper_err

def load_naidu_2025_specz(redshift):
    if (redshift < 14) or (redshift > 15):
        return None
    else:
        Muv =  -20.5
        Muv_err = 0.5
        log_density = -5.36
        log_lower_err = 0.45
        log_upper_err = 0.37
        return [Muv], [Muv_err], [log_density], [log_lower_err], [log_upper_err] 

def plot_specz_constraints(redshift, ax=None, **kwargs):
    x,y,ylo,yup = load_harikane2023_specz(redshift=redshift, type=0)
    ax.errorbar( x, y, yerr=(ylo, yup), **kwargs)
    x,y,ylo,yup = load_harikane2023_specz(redshift=redshift, type=1)
    ax.errorbar( x, y, yerr=(ylo, yup), lolims=True, **kwargs)
    ax.errorbar( x, y, yerr=(ylo, yup), **kwargs)
    x,y,ylo,yup = load_harikane2023_specz(redshift=redshift, type=2)
    ylo = np.array([0.3 for t in ylo if t==0])
    ax.errorbar( x, y, yerr=(ylo, yup), uplims=True, **kwargs)
    ax.errorbar( x, y, yerr=(ylo, yup), **kwargs)
    #
    if (redshift >= 14) and (redshift <= 15):
        x,xe,y,ylo,yup = load_naidu_2025_specz(redshift=redshift)
        ax.errorbar( x, y, xerr=xe, yerr=(ylo, yup), **kwargs)

def load_harikane_2023_photoz_digit(redshift):
    """
    Load the data from Harikane et al. 2023 
    """
    f = np.genfromtxt('obdata/UVLF/Harikane2023_Photoz_digit.dat', names=True)
    select = (f['z'] == redshift)
    log_phi_err = (f["logPhi"][select] - f["lo"][select],  f["hi"][select] - f["logPhi"][select])
    mag_err     = (f["ABmag"][select] - f["left"][select],  f["right"][select] - f["ABmag"][select])

    return f['ABmag'][select], f["logPhi"][select], mag_err, log_phi_err

def plot_photoz_constraints(redshift, ax=None, **kwargs):
    basepath = "./obdata/UVLF/UVLF_photoz/"

    ########################################################
    f = np.genfromtxt(basepath + "Bouwens2021+Oesch2018.dat" , names=True)
    sel = (f['z'] == redshift) & (f['Phi'] - f['dPhi'] > 0)
    x = f['Muv'][sel]; y = np.log10(f['Phi'][sel])  
    yup = np.log10(f['Phi'][sel] + f['dPhi'][sel])  
    ylo = np.log10(f['Phi'][sel] - f['dPhi'][sel])
    ax.errorbar(x, y, yerr=(y-ylo, yup-y), **kwargs)

    sel = (f['z'] == redshift) & (f['Phi'] - f['dPhi'] == 0)
    x = f['Muv'][sel]; y = np.log10(f['Phi'][sel])  
    yup = np.log10(f['Phi'][sel] + f['dPhi'][sel]) 
    ax.errorbar(x, y, yerr=(0.3 * np.ones(len(y)), yup-y), uplims=True, **kwargs)

    ########################################################
    f = np.genfromtxt(basepath + "Bowler2020.dat" , names=True)
    sel = f['z'] == redshift
    x = f['Muv'][sel]; y = np.log10(f['Phi'][sel])
    yup = np.log10(f['Phi'][sel] + f['up'][sel])
    ylo = np.log10(f['Phi'][sel] - f['lo'][sel])
    ax.errorbar(x, y, xerr=f['dMuv'][sel], yerr=(y-ylo, yup-y), **kwargs)

    ########################################################
    f = np.genfromtxt(basepath + "Castellano2023.dat" , names=True)
    sel = (f['z'] == redshift) & (f['type'] == 0)
    x = f['Muv'][sel]; y = np.log10(f['Phi'][sel])
    yup = np.log10(f['Phi'][sel] + f['up'][sel])
    ylo = np.log10(f['Phi'][sel] - f['lo'][sel])
    ax.errorbar(x, y, xerr=f['dMuv'][sel], yerr=(y-ylo, yup-y), **kwargs)

    sel = (f['z'] == redshift) & (f['type'] == 2)
    x = f['Muv'][sel]; y = np.log10(f['Phi'][sel])
    yup = np.log10(f['Phi'][sel] + f['up'][sel])
    ax.errorbar(x, y, xerr=f['dMuv'][sel], yerr=(0.3*np.ones(len(y)), yup-y), uplims=True, **kwargs)

    ########################################################
    f = np.genfromtxt(basepath + "Donnan2023.dat" , names=True)
    sel = f['z'] == redshift
    x = f['Muv'][sel]; y = np.log10(f['Phi'][sel])
    yup = np.log10(f['Phi'][sel] + f['up'][sel])
    ylo = np.log10(f['Phi'][sel] - f['lo'][sel])
    ax.errorbar(x, y, xerr=f['dMuv2'][sel]/2., yerr=(y-ylo, yup-y), **kwargs)

    ########################################################
    f = np.genfromtxt(basepath + "Finkelstein2022a.dat" , names=True)
    if redshift == 10:
        y = np.log10(f['Phi'])
        yup = np.log10(f['Phi'] + f['up']); ylo = np.log10(f['Phi'] - f['lo'])
        ax.fill_between(f['Muv'], ylo, yup, color='gray', edgecolor='gray', alpha=0.5)

    ########################################################
    f = np.genfromtxt(basepath + "Finkelstein2022b.dat" , names=True)
    sel = f['z'] == redshift
    x = f['Muv'][sel]; y = np.log10(f['Phi'][sel])
    yup = np.log10(f['Phi'][sel] + f['up'][sel])
    ylo = np.log10(f['Phi'][sel] - f['lo'][sel])
    ax.errorbar(x, y, yerr=(y-ylo, yup-y), **kwargs)

    ########################################################
    f = np.genfromtxt(basepath + "Harikane2022.dat" , names=True)
    sel = f['z'] == redshift
    x = f['Muv'][sel]; y = np.log10(f['Phi'][sel])
    yup = np.log10(f['Phi'][sel] + f['up'][sel])
    ylo = np.log10(f['Phi'][sel] - f['lo'][sel])
    ax.errorbar(x, y, xerr=f['dMuv'][sel], yerr=(y-ylo, yup-y), **kwargs)

    ########################################################
    f = np.genfromtxt(basepath + "Harikane2023.dat" , names=True)
    sel = (f['z'] == redshift) & (f['type'] == 0)
    x = f['Muv'][sel]; y = np.log10(f['Phi'][sel])
    yup = np.log10(f['Phi'][sel] + f['up'][sel])
    ylo = np.log10(f['Phi'][sel] - f['lo'][sel])
    ax.errorbar(x, y, xerr=f['dMuv'][sel], yerr=(y-ylo, yup-y), **kwargs)

    sel = (f['z'] == redshift) & (f['type'] == 2)
    x = f['Muv'][sel]; y = np.log10(f['Phi'][sel])
    yup = np.log10(f['Phi'][sel] + f['up'][sel])
    ax.errorbar(x, y, xerr=f['dMuv'][sel], yerr=(0.3*np.ones(len(y)), yup-y), uplims=True, **kwargs)

    ########################################################
    f = np.genfromtxt(basepath + "McLeod2016.dat" , names=True)
    sel = f['z'] == redshift
    x=f['Muv'][sel]; y=f['logPhi'][sel]
    yup = f['up'][sel]; ylo = f['lo'][sel] 
    xl  = f['left'][sel]; xr = f['right'][sel]
    ax.errorbar(x, y, xerr=(x-xl, xr-x), yerr=(y-ylo, yup-y), **kwargs)

    ########################################################
    f = np.genfromtxt(basepath + "Morishita2018+2023.dat" , names=True)
    sel = f['z'] == redshift
    x = f['Muv'][sel]; y=f['logPhi'][sel]; dyup=f['up'][sel]; dylo=f['lo'][sel]
    ax.errorbar(x, y, yerr=(dylo, dyup), **kwargs)

    ########################################################
    f = np.genfromtxt(basepath + "Naidu2023.dat" , names=True)
    sel = f['z'] == redshift
    x = f['Muv'][sel]; y=f['logPhi'][sel]; dyup=f['up'][sel]; dylo=f['lo'][sel]
    ax.errorbar(x, y, xerr=f["dMuv"][sel], yerr=(dylo, dyup), **kwargs)

    ########################################################
    f = np.genfromtxt(basepath + "Perez2023.dat" , names=True)
    sel = f['z'] == redshift
    x = f['Muv'][sel]; y=f['logPhi'][sel]; dyup=f['up'][sel]; dylo=f['lo'][sel]
    ax.errorbar(x, y, yerr=(dylo, dyup), **kwargs)

    ########################################################
    f = np.genfromtxt(basepath + "Stefanon2019.dat" , names=True)
    sel = f['z'] == redshift
    x = f['Muv'][sel]; y = np.log10(f['Phi'][sel])
    yup = np.log10(f['Phi'][sel] + f['up'][sel])
    ylo = np.log10(f['Phi'][sel] - f['lo'][sel])
    ax.errorbar(x, y, yerr=(y-ylo, yup-y), **kwargs)

    ########################################################
    f = np.genfromtxt(basepath + "McLeod2023.dat" , names=True)
    sel = f['z'] == redshift
    x = f['Muv'][sel]; y = np.log10(f['Phi'][sel])
    yup = np.log10(f['Phi'][sel] + f['dPhi'][sel])
    ylo = np.log10(f['Phi'][sel] - f['dPhi'][sel])
    ax.errorbar(x, y, yerr=(y-ylo, yup-y), **kwargs)

    ########################################################
    f = np.genfromtxt(basepath + "Adams2023.dat" , names=True)
    sel = f['z'] == redshift
    x = f['Muv'][sel]; y = np.log10(f['Phi'][sel])
    yup = np.log10(f['Phi'][sel] + f['dPhi'][sel])
    ylo = np.log10(f['Phi'][sel] - f['dPhi'][sel])
    ax.errorbar(x, y, yerr=(y-ylo, yup-y), **kwargs)

    ########################################################
    f = np.genfromtxt(basepath + "Donnan2024.dat" , names=True)
    sel = f['z'] == redshift
    x = f['Muv'][sel]; y = np.log10(f['Phi'][sel]) + np.log10(1e-6)
    yup = np.log10(f['Phi'][sel] + f['up'][sel]) + np.log10(1e-6)
    lower_phi = f['Phi'][sel] - f['lo'][sel]
    lower_phi[lower_phi <= 0] = 1e-300
    ylo = np.log10(lower_phi) + np.log10(1e-6)
    # ylo = np.log10(f['Phi'][sel] - f['lo'][sel]) + np.log10(1e-6)
    ax.errorbar(x, y, xerr=f['dMuv2'][sel]/2., yerr=(y-ylo, yup-y), **kwargs)

    ########################################################
    f = np.genfromtxt(basepath + "Robertson2024.dat" , names=True)
    sel = f['z'] == redshift
    x = f['Muv'][sel]; y = np.log10(f['Phi'][sel]) + np.log10(1e-4)
    xerrup = f['dMuv_up'][sel]
    xerrlo = f['dMuv_lo'][sel]
    yup = np.log10(f['Phi'][sel] + f['dPhi'][sel]) + np.log10(1e-4)
    ylo = np.log10(f['Phi'][sel] - f['dPhi'][sel]) + np.log10(1e-4)
    ax.errorbar(x, y, xerr=(xerrlo, xerrup), yerr=(y-ylo, yup-y), **kwargs)

    ########################################################
    f = np.genfromtxt(basepath + "Casey2024.dat" , names=True)
    sel = f['z'] == redshift
    x = f['Muv'][sel]; y = np.log10(f['Phi'][sel]) + np.log10(1e-6)
    xerr = f['dMuv2'][sel]/2.
    yup = np.log10(f['Phi'][sel] + f['dPhi'][sel]) + np.log10(1e-6)
    ylo = np.log10(f['Phi'][sel] - f['dPhi'][sel]) + np.log10(1e-6)
    ax.errorbar(x, y, xerr=xerr, yerr=(y-ylo, yup-y), **kwargs)

    f = np.genfromtxt(basepath + "Whitler2025.dat" , names=True)
    sel = f['z'] == redshift
    x = f['Muv'][sel]; y = np.log10(f['Phi'][sel]) + np.log10(1e-5)
    xerr = f['dMuv2'][sel]/2.
    yup = np.log10(f['Phi'][sel] + f['up'][sel]) + np.log10(1e-5)
    ylo = np.log10(f['Phi'][sel] - f['lo'][sel]) + np.log10(1e-5)
    ax.errorbar(x, y, xerr=xerr, yerr=(y-ylo, yup-y), **kwargs)


## HST constraints compiled in Vogelsberger+2020
def mv2020_plt_obdata(fname, ax=None, **kwargs):
	obdataPath = "obdata/UVLF/obdata_MV2020_archived/"
	obdata=np.genfromtxt(obdataPath+fname,names=True, comments='#')
	x_ob=obdata['m']
	phi=obdata['phi']
	id1=phi>0
	id2= (phi<=0)
	y_ob=0.0*x_ob
	y_ob[id1]=np.log10(phi[id1])
	y_ob[id2]=phi[id2]
	uperr=0.0*x_ob
	uperr[id1]=np.log10(phi[id1]+obdata['uperr'][id1])-y_ob[id1]
	uperr[id2]=obdata['uperr'][id2]
	low=phi-obdata['lowerr']
	low[low<=0]=1e-10
	lowerr=0.0*x_ob
	lowerr[id1]=-np.log10(low[id1])+y_ob[id1]
	lowerr[id2]=obdata['lowerr'][id2]
	
	ax.errorbar(x_ob,y_ob,yerr=(lowerr,uperr), **kwargs)

def plot_preJWST_constraints(redshift, ax=None, **kwargs):
    if redshift == 4:
        mv2020_plt_obdata('obdata'+str(21).zfill(3)+'.dat', ax=ax, **kwargs)
        #mv2020_plt_obdata('obdata'+str(21).zfill(3)+'_ono.dat', ax=ax, **kwargs) # not consistent with other observations
        mv2020_plt_obdata('obdata'+str(21).zfill(3)+'_par.dat', ax=ax, **kwargs)
    elif redshift == 5:
        mv2020_plt_obdata('obdata'+str(17).zfill(3)+'.dat', ax=ax, **kwargs)
        mv2020_plt_obdata('obdata'+str(17).zfill(3)+'_ono.dat', ax=ax, **kwargs)
    elif redshift==7:
        mv2020_plt_obdata('obdata'+str(11).zfill(3)+'.dat', ax=ax, **kwargs)
        mv2020_plt_obdata('obdata'+str(11).zfill(3)+'_ouc.dat', ax=ax, **kwargs)
        mv2020_plt_obdata('obdata'+str(11).zfill(3)+'_ate.dat', ax=ax, **kwargs)
    elif redshift==6:
        mv2020_plt_obdata('obdata'+str(13).zfill(3)+'.dat', ax=ax, **kwargs)
        mv2020_plt_obdata('obdata'+str(13).zfill(3)+'_bou.dat', ax=ax, **kwargs)
        mv2020_plt_obdata('obdata'+str(13).zfill(3)+'_ate.dat', ax=ax, **kwargs)
    elif redshift == 8:
        mv2020_plt_obdata('obdata'+str(8).zfill(3)+'.dat', ax=ax, **kwargs)
    elif redshift == 9:
        mv2020_plt_obdata('obdata'+str(4).zfill(3)+'.dat', ax=ax, **kwargs)