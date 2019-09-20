import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, trapz, simps
from scipy.optimize import fsolve
from math import factorial, pi
import pickle
# 常数定义
def go(dm,Mnum):
    #将go定义在一个函数中，方便重复调用（计算不同的吸积率和黑洞质量）
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint, trapz, simps
    from scipy.optimize import fsolve
    from math import factorial, pi

    # 求表面温度

    def Bv_Ts(miu, Ts):
        result = 4 * pi * (2 * planck * miu ** 3) / (c ** 2) / (np.exp(planck * miu / kb / Ts) - 1)
        return result

    def F__v(T_test, i):
        miu_range = np.logspace(10, 20)
        F_v = 2 * (1 - np.exp(-2 * tv_xing_sorted_array[i])) / (
                    1 + ((kabs_sorted_array[i] + kes) / kabs_sorted_array[i]) ** (1 / 2)) * pi * Bv_Ts(miu_range,
                                                                                                       T_test)
        fth = np.exp(-(np.log(kb * T_test / planck / miu_range)) / (tv_A_sorted_array[i] ** 2 * np.log(
            1 + 4 * kb * T_test / me / c / c + 16 * (kb * T_test / me / c / c) ** 2)))
        for j in range(len(fth)):
            if fth[j] > 1:
                fth[j] = 1
        Fc = trapz(fth * F_v * 3 * kb * T_test / planck / miu_range, miu_range)
        C = 4 * Fc / a / c / (T_test ** 4)
        fv = 2 * (1 - np.exp(-2 * tv_xing_sorted_array[i])) * (1 - fth) / (1 + (
                    (tv_A_sorted_array[i] / sigma_sorted_array[i] + kes) / tv_A_sorted_array[i] * sigma_sorted_array[
                i]) ** (1 / 2)) + C
        Fv = pi * Bv_Ts(miu_range, T_test) * fv
        F = trapz(Fv, miu_range)
        result = (F - Fz_sorted_array[i] / 8)
        return result

    def qiuTs():
        result = []
        for i in range(len(r_sorted_array)):
            result.append(float(fsolve(F__v, T_sorted_array[i], args=(i))))
            if i % 100 == 0:
                print(i)
        return np.array(result)

    # 求各种能量

    def Egrav():
        Eg_released = []
        for i in range(len(r_sorted_array) - 1):
            Eg_released.append(
                G * Mnum * M0 * dm * dmc / r_sorted_array[i] / rg - G * Mnum * M0 * dm * dmc / r_sorted_array[
                    i + 1] / rg)
        return Eg_released

    def Evis():
        Evis_released = []
        for i in range(len(r_sorted_array) - 1):
            Evis_released.append(trapz([Qvis_sorted_array[i], Qvis_sorted_array[i + 1]],
                                       [r_sorted_array[i] * rg, r_sorted_array[i + 1] * rg]))
        return Evis_released

    def Erad():
        Erad_released = []
        for i in range(len(r_sorted_array) - 1):
            Erad_released.append(trapz([Qrad_sorted_array[i], Qrad_sorted_array[i + 1]],
                                       [r_sorted_array[i] * rg, r_sorted_array[i + 1] * rg]))
        return Erad_released

    # 绘图

    def fig():
        plt.subplot(2, 2, 1)
        plt.loglog(r_array, T_array, marker='.', linestyle='')
        plt.loglog(r_sorted_array, T_sorted_array, marker='.', linestyle='')
        plt.ylabel('T/K')
        plt.subplot(2, 2, 2)
        plt.loglog(r_array, l_lk_array, marker='.', linestyle='')
        plt.ylabel('l/lk')
        plt.subplot(2, 2, 3)
        plt.loglog(r_array, v_c_array, marker='.', linestyle='')
        plt.loglog(r_array, vs_c_array, marker='.', linestyle='')
        plt.ylabel('v/c')
        plt.subplot(2, 2, 4)
        plt.loglog(r_array, np.array(H_array) / np.array(r_array) / rg, marker='.', linestyle='')
        plt.ylabel('H/r')

    # 求辐射
    def Bv(miu):
        result = 4 * pi * (2 * planck * miu ** 3) / (c ** 2) / (np.exp(planck * miu / kb / T_sorted_array) - 1)
        return result

    def Iv(miu):
        result = Bv(miu)
        return result

    def Fv(miu):
        result = trapz(Iv(miu) * 2 * pi * rg * r_sorted_array, rg * r_sorted_array)
        return result

    def LA(A):
        miu = c/(A*10**(-8))
        result = miu*Fv(miu)
        return result*2

    def L(miu_min, miu_max):
        miu_range = np.logspace(miu_min, miu_max,100)
        F = []
        for i in range(len(miu_range)):
            F.append(Fv(miu_range[i]))
        return np.array(F) * 2, miu_range

    def L2():
        result = trapz(4 * pi * 2 * pi * r_sorted_array * rg * (2 * pi ** 4 * kb ** 4 * T_sorted_array ** 4) / 15 / (
                    planck ** 3) / c / c, r_sorted_array * rg)
        return result * 2

    # 求解龙格库塔

    def qiu_T(x, a, b, c):
        ansT = a * x ** 4 + b * x + c
        return ansT

    def runge(y, r):
        l_tem = y[0]
        yita_tem = y[1]
        omigak = ((G * M / r / rg) ** (1 / 2)) / (r - 1) / rg
        lk = omigak * ((r * rg) ** 2) / rg / c
        W = dm * dmc * c * rg * (l_tem - lin) / 2 / pi / alpha / ((r * rg) ** 2)
        sigma = ((dm * dmc) ** 2) / ((2 * pi * r * rg) ** 2) / W / yita_tem
        v = -dm * dmc / 2 / pi / r / rg / sigma
        K = W / sigma
        H = ((2 * (N + 1) * W * IN / sigma / IN_1) ** (1 / 2)) / omigak
        yaqiang = W / 2 / H / IN_1
        midu = sigma / 2 / H / IN
        vs = (yaqiang / midu) ** (1 / 2)
        T_tem = float(fsolve(qiu_T, 1000, (a / 3, Rg * midu / miu, -yaqiang)))

        kabs = 0.64 * 10 ** (23) * midu * T_tem ** (-7 / 2)
        teff = (sigma * (kes + kabs) * kabs * sigma) ** (1 / 2)
        keff = teff / sigma
        kk = kabs + kes
        Fz = (8 * a * c * T_tem ** 4) / (3 * kk * midu * H)
        beita = Rg * midu * T_tem / miu / yaqiang
        T1 = (32 - 24 * beita - 3 * beita ** 2) / (24 - 21 * beita)
        T3 = (32 - 27 * beita) / (24 - 21 * beita)
        d_omigak_dr = (-3 / 2) / (r * rg)

        l_yita = [((1 / c) * ((T1 + 1) / (T3 - 1) / r / rg + (T1 - 1) / (
                T3 - 1) * d_omigak_dr + 4 * pi * r * rg * Fz / K / dm / dmc - 4 * pi * alpha * W * l_tem * c / K / dm / dmc / r - (
                                      3 * T1 - 1) / (2 * yita_tem * (T3 - 1)) * (((
                                                                                          l_tem ** 2 - lk ** 2) * c ** 2 / r ** 3 / K / rg) - yita_tem / r / rg - d_omigak_dr + 2 / r / rg * (
                                                                                         1 + yita_tem))) / (
                           dm * dmc * T1 / ((rg * r) ** 2) / pi / alpha / W / (
                           T3 - 1) - 2 * pi * alpha * W / K / dm / dmc - (
                                   (1 + yita_tem) * dm * dmc / ((r * rg) ** 2) / 2 / pi / alpha / W * (
                                   3 * T1 - 1) / 2 / yita_tem / (T3 - 1)))),
                  rg * ((((
                                      l_tem ** 2 - lk ** 2) * c ** 2 / r ** 3 / K / rg) - yita_tem / r / rg - d_omigak_dr + 2 / r / rg * (
                                     1 + yita_tem)) - c * (
                                1 + yita_tem) * dm * dmc / 2 / pi / alpha / W / ((r * rg) ** 2) * ((1 / c) * (
                          (T1 + 1) / (T3 - 1) / r / rg + (T1 - 1) / (
                          T3 - 1) * d_omigak_dr + 4 * pi * r * rg * Fz / K / dm / dmc - 4 * pi * alpha * W * l_tem * c / K / dm / dmc / r - (
                                  3 * T1 - 1) / (2 * yita_tem * (T3 - 1)) * (
                                  ((
                                               l_tem ** 2 - lk ** 2) * c ** 2 / r ** 3 / K / rg) - yita_tem / r / rg - d_omigak_dr + 2 / r / rg * (
                                          1 + yita_tem))) / (dm * dmc * T1 / ((rg * r) ** 2) / pi / alpha / W / (
                          T3 - 1) - 2 * pi * alpha * W / K / dm / dmc - ((1 + yita_tem) * dm * dmc / (
                          (r * rg) ** 2) / 2 / pi / alpha / W * (3 * T1 - 1) / 2 / yita_tem / (T3 - 1)))))]

        dl_dr = c * l_yita[0]
        dyita_dr = l_yita[1] / rg

        xing1 = (l_tem ** 2 - lk ** 2) * c * c * rg * rg / ((r * rg) ** 3) / K - d_omigak_dr + T1 / r / rg + (
                    T1 - 1) * (
                        (IN * (N + 1) * K / IN_1 / ((H * omigak) ** 2)) * (
                            (l_tem ** 2 - lk ** 2) * c * c * rg * rg / K / (
                            (r * rg) ** 3) - d_omigak_dr + 1 / r / rg) - d_omigak_dr)
        xing2 = 4 * pi * r * rg * alpha * W / dm / dmc + 2 * pi * r * r * rg * rg * alpha * W * (
                (l_tem ** 2 - lk ** 2) * c * c * rg * rg / r / r / r / rg / rg / rg / K - d_omigak_dr) / dm / dmc
        xing3 = T1 - yita_tem + (T1 - 1) * (1 - yita_tem) * IN * (N + 1) * K / IN_1 / ((H * omigak) ** 2)

        d_lnvr_dr = (dm * dmc * W * xing1 / sigma / (
                T3 - 1) - 4 * pi * r * rg * Fz + 4 * pi * l_tem * c * rg * alpha * W / r / rg - 2 * pi * alpha * W * xing2) / (
                            -dm * dmc * W * xing3 / sigma / (
                            T3 - 1) - 2 * pi * alpha * W * 2 * pi * r * r * rg * rg * alpha * W * yita_tem / dm / dmc)
        d_lnW_dr = (dl_dr - 4 * pi * r * rg * alpha * W / dm / dmc) / (2 * pi * r * r * rg * rg * alpha * W / dm / dmc)
        d_lnsigma_dr = -1 / r / rg - d_lnvr_dr
        d_lnh_dr = (IN * (N + 1) * K / IN_1 / ((H * omigak) ** 2)) * (
                (l_tem ** 2 - lk ** 2) * c * rg * c * rg / K / ((r * rg) ** 3) - d_omigak_dr + 1 / r / rg + (
                1 - yita_tem) * d_lnvr_dr) - d_omigak_dr
        Qvis = -2 * pi * alpha * W * dl_dr + 4 * l_tem * c * rg * pi * alpha * W / r / rg
        Qrad = 4 * pi * r * rg * Fz

        Qadv = -dm * dmc * W / sigma / (T3 - 1) * (d_lnW_dr - T1 * d_lnsigma_dr + (T1 - 1) * d_lnh_dr)


        r_array.append(r)
        l_c_rg_array.append(l_tem / c / rg)
        v_c_array.append(abs(v / c))
        vs_c_array.append(vs / c)
        T_array.append(T_tem)
        l_lk_array.append(l_tem / lk)
        midu_array.append(midu)
        yaqiang_array.append(yaqiang)
        H_array.append(H)
        kabs_array.append(kabs)
        keff_array.append(keff)
        Qadv_array.append(Qadv)
        Qrad_array.append(Qrad)
        Qvis_array.append(Qvis)
        sigma_array.append(sigma)
        Fz_array.append(Fz)

        return l_yita

    # 常数定义

    kb = 1.3806505 * 10 ** (-16)
    planck = 6.62606896 * 10 ** (-27)
    G = 6.67259 * 10 ** (-8)
    c = 3 * 10 ** 10
    kes = 0.34
    me = 9.1 * 10 ** (-28)
    mp = 1.672621637 * 10 ** (-24)
    thomson = 6.6524 * 10 ** (-25)
    M0 = 1.9891 * 10 ** 33
    alpha = 0.1
    miu = 0.617
    Rg = 8.314 * (10 ** 7)
    a = 7.5657 * 10 ** (-15)

    firstflag = 0
    lin_max = 0
    lin_min = 0
    minflag = 0
    maxflag = 0
    r_v_vs_array_sorted = [[100, 0.5]]

    while ((firstflag == 0) | (min(r_v_vs_array_sorted)[0] > 3) | (min(r_v_vs_array_sorted)[1] < 1)):

        # 可变参数调节

        #Mnum = 10000000
        #dm = 100

        if firstflag == 0:
            lin = 1.68
            firstflag = 1
        elif (maxflag == 1 & minflag == 1):
            lin = (lin_max + lin_min) / 2

        # 参数定义

        M = Mnum * M0
        rg = 2 * G * M / (c ** 2)
        dmc = 1.7 * 10 ** 17 * (M / M0)

        # 结果记录

        r_array = []
        l_c_rg_array = []
        v_c_array = []
        vs_c_array = []
        T_array = []
        l_lk_array = []
        midu_array = []
        yaqiang_array = []
        H_array = []
        kabs_array = []
        keff_array = []
        r_v_vs_array = []
        Qrad_array = []
        Qadv_array = []
        Qvis_array = []
        sigma_array = []
        Fz_array = []

        # 边界值定义

        N = 3
        IN = ((2 ** N * factorial(N)) ** 2) / (factorial(2 * N + 1))
        IN_1 = ((2 ** (N + 1) * factorial(N + 1)) ** 2) / (factorial(2 * N + 3))

        r = 10000
        omigak = ((G * M / r / rg) ** (1 / 2)) / (r - 1) / rg
        l_0 = (omigak * (r * rg) ** 2) / rg / c
        v_0 = -(5.4 * 10 ** 5) * alpha ** (4 / 5) * (M / M0) ** (-1 / 5) * dm ** (3 / 10) * r ** (-1 / 4)
        sigma_0 = -dm * dmc / 2 / pi / r / rg / v_0
        W_0 = dm * dmc * c * rg * (l_0 - lin) / 2 / pi / alpha / ((r * rg) ** 2)
        yita_0 = (v_0 ** 2) * sigma_0 / W_0
        y0 = [l_0, yita_0]

        r_array0 = np.linspace(3, 10000, 9998)[::-1]
        y = odeint(runge, y0, r_array0, mxstep=10000, atol=10 ** (-12), rtol=10 ** (-12))

        for i in range(len(r_array)):
            r_v_vs_array.append([r_array[i], v_c_array[i] / vs_c_array[i]])

        r_v_vs_array_sorted = sorted(r_v_vs_array, key=(lambda x: x[0]))

        print(min(r_v_vs_array_sorted))
        print(lin, lin_max, lin_min)
        if (min(r_v_vs_array_sorted)[0] > 3):
            lin_max = lin
            maxflag = 1
            firstflag = 1
            lin = lin - 0.3
        elif (min(r_v_vs_array_sorted)[1] < 1):
            lin_min = lin
            minflag = 1
            firstflag = 1
            lin = lin + 0.3

    r_array = np.array(r_array)
    l_c_rg_array = np.array(l_c_rg_array)
    v_c_array = np.array(v_c_array)
    vs_c_array = np.array(vs_c_array)
    T_array = np.array(T_array)
    l_lk_array = np.array(l_lk_array)
    midu_array = np.array(midu_array)
    yaqiang_array = np.array(yaqiang_array)
    H_array = np.array(H_array)
    r_v_vs_array = np.array(r_v_vs_array)
    kabs_array = np.array(kabs_array)
    Qrad_array = np.array(Qrad_array)
    Qvis_array = np.array(Qvis_array)
    Qadv_array = np.array(Qadv_array)
    keff_array = np.array(keff_array)
    sigma_array = np.array(sigma_array)
    Fz_array = np.array(Fz_array)

    r_T_kabs_array = []
    for i in range(len(r_array)):
        r_T_kabs_array.append(
            [r_array[i], l_c_rg_array[i], v_c_array[i], vs_c_array[i], T_array[i], l_lk_array[i], midu_array[i],
             yaqiang_array[i], H_array[i], 0, kabs_array[i], Qrad_array[i], Qvis_array[i], Qadv_array[i], keff_array[i],
             sigma_array[i], Fz_array[i]])
    r_T_kabs_array = np.array(sorted(r_T_kabs_array, key=(lambda x: x[0])))

    r_sorted_array = r_T_kabs_array[:, 0]
    l_c_rg_sorted_array = r_T_kabs_array[:, 1]
    v_c_sorted_array = r_T_kabs_array[:, 2]
    vs_c_sorted_array = r_T_kabs_array[:, 3]
    T_sorted_array = r_T_kabs_array[:, 4]
    l_lk_sorted_array = r_T_kabs_array[:, 5]
    midu_sorted_array = r_T_kabs_array[:, 6]
    yaqiang_sorted_array = r_T_kabs_array[:, 7]
    H_sorted_array = r_T_kabs_array[:, 8]
    r_v_vs_sorted_array = r_T_kabs_array[:, 9]
    kabs_sorted_array = r_T_kabs_array[:, 10]
    Qrad_sorted_array = r_T_kabs_array[:, 11]
    Qvis_sorted_array = r_T_kabs_array[:, 12]
    Qadv_sorted_array = r_T_kabs_array[:, 13]
    keff_sorted_array = r_T_kabs_array[:, 14]
    sigma_sorted_array = r_T_kabs_array[:, 15]
    Fz_sorted_array = r_T_kabs_array[:, 16]

    tv_xing_sorted_array = keff_sorted_array * sigma_sorted_array
    tv_A_sorted_array = (sigma_sorted_array * kes + tv_xing_sorted_array) / (1 + tv_xing_sorted_array)

    Total_Egrav = G * Mnum * M0 * dm * dmc / 3 / rg - G * Mnum * M0 * dm * dmc / 10000 / rg


    Ts = (Fz_sorted_array / a / c) ** (1 / 4)
    T_sorted_array = Ts
    Lv, miu_range = L(5,20)
    efficiency = L2() / dm / dmc / c / c


    return efficiency,r_sorted_array,T_sorted_array,H_sorted_array,l_lk_sorted_array,v_c_sorted_array,rg,Mnum,L2(),Lv,miu_range,LA(5100),LA(4400),LA(3500),LA(1000),LA(3),LA(0.5)






efficiency_array = []
r_dm =[]
T_dm = []
H_dm=[]
l_lk_dm=[]
v_c_dm=[]
Mnum_dm=[]
Lbol_dm = []
Lv_dm = []
miu_range_dm = []
L_5100_dm = []
L_4400_dm = []
L_3500_dm = []
L_1000_dm = []
L_3_dm = []
L_0_5_dm = []

# 常数设定
G = 6.67259 * 10 ** (-8)
c = 3 * 10 ** 10
M0 = 1.9891 * 10 ** 33
Mnum = 10
M = Mnum * M0
rg = 2 * G * M / (c ** 2)
dmc = 1.7 * 10 ** 17 * (M / M0)

# 可变参数


Mstar = 50 * 1.9891 * 10 ** 33
tmin = Mstar/3000/dmc

# Rp_Rt = 1
# Rxing = ((tmin*((G*M)**(1/2))/(2*pi*Rp**3))**(2/3))/2
# # Rt =Rxing*(M/Mstar)**(1/3)
# # Rp = Rp_Rt * Rt


t_max = 0.000001**(-3/5)*tmin
t_range = np.logspace(np.log10(tmin), np.log10(t_max),14)
dm_range = (t_range/tmin)**(-5/3) *1000
#dm_range = np.logspace(-3,3,7)

# 撕碎的恒星质量 1

# dm_range = np.linspace(-3,3,18)
# dm_range = 10**dm_range
dm_range_str = []
for i in range(len(dm_range)):
    dm_range_str.append(str(dm_range[i])[0:5])

for dm in dm_range:
    result = go(dm,Mnum)
    efficiency_array.append(result[0])
    r_dm.append(result[1])
    T_dm.append(result[2])
    H_dm.append(result[3])
    l_lk_dm.append(result[4])
    v_c_dm.append(result[5])
    rg = result[6]
    Mnum_dm = result[7]
    Lbol_dm.append(result[8])
    Lv_dm.append(result[9])
    miu_range_dm.append(result[10])
    L_5100_dm.append(result[11])
    L_4400_dm.append(result[12])
    L_3500_dm.append(result[13])
    L_1000_dm.append(result[14])
    L_3_dm.append(result[15])
    L_0_5_dm.append((result[16]))

def fig2():
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

import os

if not os.path.exists("C:/Users/ZS/Desktop/毕设/result/10^"+str(int(np.log10(Mnum_dm)))+ '/'+str(Mstar/(1.9891 * 10 ** 33))):
    os.makedirs("C:/Users/ZS/Desktop/毕设/result/10^"+str(int(np.log10(Mnum_dm)))+ '/'+str(Mstar/(1.9891 * 10 ** 33)))


def save_var():
    #顺序存入变量
    save_file = open("C:/Users/ZS/Desktop/毕设/result/10^"+str(int(np.log10(Mnum_dm)))+ '/'+str(Mstar/(1.9891 * 10 ** 33))+"/save.bin","wb")
    pickle.dump(efficiency_array,save_file)
    pickle.dump(r_dm,save_file)
    pickle.dump(T_dm,save_file)
    pickle.dump(H_dm,save_file)
    pickle.dump(l_lk_dm,save_file)
    pickle.dump(v_c_dm,save_file)
    pickle.dump(rg,save_file)
    pickle.dump(Mnum_dm,save_file)
    pickle.dump(Lbol_dm, save_file)
    pickle.dump(Lv_dm, save_file)
    pickle.dump(miu_range_dm, save_file)
    pickle.dump(L_5100_dm, save_file)
    pickle.dump(L_4400_dm, save_file)
    pickle.dump(L_3500_dm, save_file)
    pickle.dump(L_1000_dm, save_file)
    pickle.dump(Mstar, save_file)
    pickle.dump(t_range, save_file)
    pickle.dump(t_max, save_file)
    pickle.dump(dm_range, save_file)
    pickle.dump(dm_range_str, save_file)

    save_file.close()



def load_var(Mnum, Mstar):
    # 顺序导出变量
    load_file = open("C:/Users/ZS/Desktop/毕设/result/10^" + str(Mnum) + '/' + str(Mstar) + "/save.bin", "rb")
    efficiency_array = pickle.load(load_file)  
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


save_var()
fig2()
