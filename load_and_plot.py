#此程序用于读取保存的.bin数据文件，并进行绘图


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, trapz, simps
from scipy.optimize import fsolve
from math import factorial, pi
import pickle



def load_var(Mnum, Mstar):
    # loading .bin data

    load_file = open("C:/Users/ZS/Desktop/毕设/result/10^" + str(Mnum) + '/' + str(Mstar) + "/save.bin", "rb")
    efficiency_array = pickle.load(load_file)  # 顺序导出变量
    r_dm = pickle.load(load_file)
    T_dm = pickle.load(load_file)
    H_dm = pickle.load(load_file)
    l_lk_dm = pickle.load(load_file)
    v_c_dm = pickle.load(load_file)
    rg = pickle.load(load_file)
    Mnum_dm = pickle.load(load_file)
    Lbol_dm = pickle.load(load_file)
    Lv_dm = pickle.load(load_file)
    miu_range_dm = pickle.load(load_file)
    L_5100_dm = pickle.load(load_file)
    L_4400_dm = pickle.load(load_file)
    L_3500_dm = pickle.load(load_file)
    L_1000_dm = pickle.load(load_file)
    Mstar = pickle.load(load_file)
    t_range = pickle.load(load_file)
    t_max = pickle.load(load_file)
    dm_range = pickle.load(load_file)
    dm_range_str = pickle.load(load_file)
    return {'efficiency_array': efficiency_array, 'r_dm': r_dm, 'T_dm': T_dm, 'H_dm': H_dm, 'l_lk_dm': l_lk_dm,
            'v_c_dm': v_c_dm, 'rg': rg, 'Mnum_dm': Mnum_dm, 'Lbol_dm': Lbol_dm, 'Lv_dm': Lv_dm,
            'miu_range_dm': miu_range_dm, 'L_5100_dm': L_5100_dm, 'L_4400_dm': L_4400_dm,
            'L_3500_dm': L_3500_dm, 'L_1000_dm': L_1000_dm, 'Mstar': Mstar, 't_range': t_range, 't_max': t_max,
            'dm_range': dm_range, 'dm_range_str': dm_range_str}

# distribute data to variables

data =load_var(7,10.0)
efficiency_array=data['efficiency_array']
r_dm=data['r_dm']
T_dm=data['T_dm']
H_dm=data['H_dm']
l_lk_dm=data['l_lk_dm']
v_c_dm=data['v_c_dm']
rg=data['rg']
Mnum_dm=data['Mnum_dm']
Lbol_dm=data['Lbol_dm']
Lv_dm=data['Lv_dm']
miu_range_dm=data['miu_range_dm']
L_5100_dm=data['L_5100_dm']
L_4400_dm=data['L_4400_dm']
L_3500_dm=data['L_3500_dm']
L_1000_dm=data['L_1000_dm']
Mstar=data['Mstar']
t_range=data['t_range']
t_max=data['t_max']
dm_range=data['dm_range']
dm_range_str=data['dm_range_str']

G = 6.67259 * 10 ** (-8)
c = 3 * 10 ** 10
M0 = 1.9891 * 10 ** 33
M = Mnum_dm * M0
rg = 2 * G * M / (c ** 2)
dmc = 1.7 * 10 ** 17 * (M / M0)
tmin = Mstar / 3000 / dmc

def fig2():
    # plotting 4 figures in the thesis

    picture_1 = plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    for i in range(len(T_dm)):
        plt.plot(np.log10(r_dm[i]), np.log10(T_dm[i]))
    plt.xlabel('$log(r)$', fontsize=15)
    plt.ylabel('$log(T_{eff}\ /\ K)$', fontsize=15)
    plt.tick_params(labelsize=13)


    plt.subplot(2, 2, 2)
    for i in range(len(T_dm)):
        plt.plot(np.log10(r_dm[i]), np.array(H_dm[i]) / np.array(r_dm[i]) / rg)
    plt.legend(dm_range_str, loc='upper right', fontsize=10)
    plt.xlabel('$log(r)$', fontsize=15)
    plt.ylabel('$H\ /\ R$', fontsize=15)
    plt.tick_params(labelsize=13)


    plt.subplot(2, 2, 3)
    for i in range(len(T_dm)):
        plt.plot(np.log10(r_dm[i]), l_lk_dm[i])
    plt.xlabel('$log(r)$', fontsize=15)
    plt.ylabel('$l\ /\ l_{k}$', fontsize=15)
    plt.tick_params(labelsize=13)


    plt.subplot(2, 2, 4)
    for i in range(len(T_dm)):
        plt.plot(np.log10(r_dm[i]), np.log10(v_c_dm[i]))
    plt.xlabel('$log(r)$', fontsize=15)
    plt.ylabel('$log (v_{r}\ /\ c)$', fontsize=15)
    plt.tick_params(labelsize=13)


    picture_1.suptitle('$M_{BH} = 10^{' + str(int(np.log10(Mnum_dm))) + '}' + 'M_{⊙}$', fontsize=20)
    plt.savefig("C:/Users/ZS/Desktop/毕设/result/10^" + str(int(np.log10(Mnum_dm)))+ '/'+str(Mstar/(1.9891 * 10 ** 33)) + "/figure1.png",bbox_inches='tight')

    picture_2 = plt.figure(figsize=(20, 20))

    plt.subplot(3, 2, 1)
    plt.semilogx(dm_range, efficiency_array, marker='o')
    plt.xlabel('$\dot{m}$', fontsize=25)
    plt.ylabel('$Efficiency$', fontsize=25)
    plt.tick_params(labelsize=20)

    plt.subplot(3, 2, 2)
    plt.loglog(dm_range, Lbol_dm, marker='o')
    plt.xlabel('$\dot{m}$', fontsize=25)
    plt.ylabel('$log\ [Luminosity\ /\ (erg/s)]$', fontsize=25)
    plt.tick_params(labelsize=20)

    plt.subplot(3, 2, 3)
    plt.loglog(dm_range, L_5100_dm, marker='o')
    plt.xlabel('$\dot{m}$', fontsize=25)
    plt.ylabel('$log[\ L\_5100\ /\ (erg/s)\ ]$', fontsize=25)
    plt.tick_params(labelsize=20)

    plt.subplot(3, 2, 4)
    plt.loglog(dm_range, L_4400_dm, marker='o')
    plt.xlabel('$\dot{m}$', fontsize=25)
    plt.ylabel('$log[\ L\_4400\ /\ (erg/s)\ ]$', fontsize=25)
    plt.tick_params(labelsize=20)

    plt.subplot(3, 2, 5)
    plt.loglog(dm_range, L_3500_dm, marker='o')
    plt.xlabel('$\dot{m}$', fontsize=25)
    plt.ylabel('$log[\ L\_3500\ /\ (erg/s)\ ]$', fontsize=25)
    plt.tick_params(labelsize=20)

    plt.subplot(3, 2, 6)
    plt.loglog(dm_range, L_1000_dm, marker='o')
    plt.xlabel('$\dot{m}$', fontsize=25)
    plt.ylabel('$log[\ L\_1000\ /\ (erg/s)\ ]$', fontsize=25)
    plt.tick_params(labelsize=20)

    # plt.subplot(2, 4, 7)
    # plt.loglog(dm_range, L_3_dm)
    # plt.xlabel('dm')
    # plt.ylabel('log[ L_3 / (erg/s) ]')
    #
    # plt.subplot(2, 4, 6)
    # plt.loglog(dm_range, L_0_5_dm)
    # plt.xlabel('dm')
    # plt.ylabel('log[ L_0.5 / (erg/s) ]')
    picture_2.suptitle('$M_{BH} = 10^{' + str(int(np.log10(Mnum_dm))) + '}' + 'M_{⊙}$', fontsize=30)

    plt.savefig("C:/Users/ZS/Desktop/毕设/result/10^" + str(int(np.log10(Mnum_dm))) + '/' + str(
        Mstar / (1.9891 * 10 ** 33)) + "/figure2.png", bbox_inches='tight')

    picture_3 = plt.figure(figsize=(20, 20))

    plt.subplot(3, 2, 1)
    plt.loglog(t_range/tmin, Lbol_dm,marker = 'o')
    plt.loglog(t_range/tmin, Lbol_dm[-1]/(Lbol_dm[0] * (t_range[-1]/tmin) ** (-5 / 3)) * Lbol_dm[0] * (t_range/tmin) ** (-5 / 3),linestyle = '--')
    plt.xlabel('$log[\ t/tmin\ ]$', fontsize=25)
    plt.ylabel('$log[\ L\_bol\ /\ (erg/s)\ ]$', fontsize=25)
    plt.legend(['Computation Result','-5/3 slope'],fontsize = 25,loc='upper right')
    plt.tick_params(labelsize=20)

    plt.subplot(3, 2, 2)
    plt.loglog(t_range/tmin, L_5100_dm,marker = 'o')
    plt.loglog(t_range/tmin, L_5100_dm[-1]/(L_5100_dm[0] * (t_range[-1]/tmin) ** (-5 / 3)) * L_5100_dm[0] * (t_range/tmin) ** (-5 / 3),linestyle = '--')
    plt.xlabel('$log[\ t/tmin\ ]$', fontsize=25)
    plt.ylabel('$log[\ L\_5100\ /\ (erg/s)\ ]$', fontsize=25)
    plt.tick_params(labelsize=20)

    plt.subplot(3, 2, 3)
    plt.loglog(t_range/tmin, L_4400_dm,marker = 'o')
    plt.loglog(t_range/tmin, L_4400_dm[-1]/(L_4400_dm[0] * (t_range[-1]/tmin) ** (-5 / 3)) * L_4400_dm[0] * (t_range/tmin) ** (-5 / 3),linestyle = '--')
    plt.xlabel('$log[\ t/tmin\ ]$', fontsize=25)
    plt.ylabel('$log[\ L\_4400\ /\ (erg/s)\ ]$', fontsize=25)
    plt.tick_params(labelsize=20)

    plt.subplot(3, 2, 4)
    plt.loglog(t_range/tmin, L_3500_dm, marker = 'o')
    plt.loglog(t_range/tmin, L_3500_dm[-1]/(L_3500_dm[0] * (t_range[-1]/tmin) ** (-5 / 3)) * L_3500_dm[0] * (t_range/tmin) ** (-5 / 3),linestyle = '--')
    plt.xlabel('$log[\ t/tmin\ ]$', fontsize=25)
    plt.ylabel('$log[\ L\_3500\ /\ (erg/s)\ ]$', fontsize=25)
    plt.tick_params(labelsize=20)

    plt.subplot(3, 2, 5)
    plt.loglog(t_range/tmin, L_1000_dm,marker = 'o')
    plt.loglog(t_range/tmin, L_1000_dm[-1]/(L_1000_dm[0] * (t_range[-1]/tmin) ** (-5 / 3)) * L_1000_dm[0] * (t_range/tmin) ** (-5 / 3),linestyle = '--')
    plt.xlabel('$log[\ t/tmin\ ]$', fontsize=25)
    plt.ylabel('$log[\ L\_1000\ /\ (erg/s)\ ]$', fontsize=25)
    plt.tick_params(labelsize=20)


    # plt.subplot(2, 4, 6)
    # plt.loglog(t_range / tmin, L_3_dm)
    # plt.loglog(t_range / tmin,L_3_dm[-1] / (L_3_dm[0] * (t_range[-1] / tmin) ** (-5 / 3)) * L_3_dm[0] * (t_range / tmin) ** (-5 / 3), linestyle='--')
    # plt.xlabel('log[ t/tmin ]')
    # plt.ylabel('log[ L_3 / (erg/s) ]')
    #
    #
    # plt.subplot(2, 4, 7)
    # plt.loglog(t_range / tmin, L_0_5_dm)
    # plt.loglog(t_range / tmin,
    #            L_0_5_dm[-1] / (L_0_5_dm[0] * (t_range[-1] / tmin) ** (-5 / 3)) * L_0_5_dm[0] * (t_range / tmin) ** (
    #                        -5 / 3), linestyle='--')
    # plt.xlabel('log[ t/tmin ]')
    # plt.ylabel('log[ L_0.5 / (erg/s) ]')


    picture_3.suptitle('$M = 10^{' + str(int(np.log10(Mnum_dm))) + '} M_{⊙}\ ——\ M_{*} = '+ str(Mstar/(1.9891 * 10 ** 33))+' M_{⊙}$',fontsize = 30)
    plt.savefig("C:/Users/ZS/Desktop/毕设/result/10^" + str(int(np.log10(Mnum_dm)))+ '/'+str(Mstar/(1.9891 * 10 ** 33)) + "/figure3.png",bbox_inches='tight')

    picture_4 = plt.figure(figsize=(15, 10))

    for i in range(len(T_dm)):
        plt.plot(np.log10(np.array(miu_range_dm[i])), np.log10(Lv_dm[i] * np.array(miu_range_dm[i])))
    plt.legend(dm_range_str, loc='upper right',fontsize = 15)
    plt.xlabel('$log(v)$',fontsize = 20)
    plt.ylabel('$log(vLv)$',fontsize = 20)
    plt.ylim((30, 48))
    plt.xlim((10, 20))
    plt.tick_params(labelsize=15)

    picture_4.suptitle('$M_{BH} = 10^{' + str(int(np.log10(Mnum_dm))) + '}' + 'M_{⊙}$', fontsize=20)
    plt.savefig("C:/Users/ZS/Desktop/毕设/result/10^" + str(int(np.log10(Mnum_dm)))+ '/'+str(Mstar/(1.9891 * 10 ** 33)) + "/figure4.png",bbox_inches='tight')

    for i in range(len(T_dm)):
        plt.plot(np.log10(np.array(miu_range_dm[i])), np.log10(Lv_dm[i] * np.array(miu_range_dm[i])))
    plt.legend(dm_range_str, loc='upper right',fontsize = 15)
    plt.xlabel('$log(v)$',fontsize = 20)
    plt.ylabel('$log(vLv)$',fontsize = 20)
    plt.ylim((30, 48))
    plt.xlim((10, 20))
    plt.tick_params(labelsize=15)

    picture_4.suptitle('$M_{BH} = 10^{' + str(int(np.log10(Mnum_dm))) + '}' + 'M_{⊙}$', fontsize=20)
    plt.savefig("C:/Users/ZS/Desktop/毕设/result/10^" + str(int(np.log10(Mnum_dm)))+ '/'+str(Mstar/(1.9891 * 10 ** 33)) + "/figure4.png",bbox_inches='tight')

fig2()