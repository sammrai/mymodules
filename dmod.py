# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 14:45:19 2016

@author: daisuke
"""

import os
import socio
import st
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def load_float1(flag, light_float, ps_float, nps_float):
    if flag == 0 :
        ### .pkl読み込み
        ps_img = st.loadpickle("./pkl/ps_img.pkl")
        print("ps_img loaded...")
        nps_img = st.loadpickle("./pkl/nps_img.pkl")
        print("nps_img loaded...")
        light_img = st.loadpickle("./pkl/light_img.pkl")
        print("light_img loaded...")
    else :
        #floatファイルをロード
        ps = socio.openfloat(ps_float)
        nps = socio.openfloat(nps_float)
        #socioのインスタンスtestからスライスの記法でロード
        ps_img = ps[:,:,:]
        print("ps_img loaded...")
        nps_img = nps[:,:,:]
        print("nps_img loaded...")
        ### .pkl（中間ファイル）に保存
        if os.path.isdir("pkl") == False :
            os.mkdir("pkl")
        st.savepickle("pkl/ps_img.pkl",ps_img)
        st.savepickle("pkl/nps_img.pkl",nps_img)
        
        ### light.floatの読み込み
        light = socio.openfloat(light_float)
        ### socioのインスタンスtestからスライスの記法でロード
        light_img = light[:,:,:]
        print("light_img loaded...")
        ### .pkl（中間ファイル）に保存
        st.savepickle("pkl/light_img.pkl",light_img)

        #波長取得
        wavelength = ps.getWavelengths()
        if os.path.isdir("results") == False :
            os.mkdir("resutls")
        np.savetxt("results/wavelength.txt", wavelength)

    return(light_img, ps_img, nps_img)

def load_float2(flag, light_float, smp_float):
    if flag == 0 :
        ### pklファイルを読み込む
        smp_img = st.loadpickle("pkl/sample_img.pkl")
        print("sample_img loaded...")
        light_img = st.loadpickle("pkl/light_img.pkl")
        print("light_img loaded...")
    else :
        #floatファイルをロード
        smp = socio.openfloat(smp_float)
        #socioのインスタンスtestからスライスの記法でロード
        smp_img = smp[:,:,:]
        print("sample_img loaded...")
        ### .pkl（中間ファイル）に保存
        if os.path.isdir("pkl") == False :
            os.mkdir("pkl")
        st.savepickle("pkl/sample_img.pkl",smp_img)
        
        ### light.floatの読み込み
        light = socio.openfloat(light_float)
        ### socioのインスタンスtestからスライスの記法でロード
        light_img = light[:,:,:]
        print("light_img loaded...")
        ### .pkl（中間ファイル）に保存
        st.savepickle("pkl/light_img.pkl",light_img)

        #波長取得
        wavelength = smp.getWavelengths()
        if os.path.isdir("results") == False :
            os.mkdir("results")
        np.savetxt("results/wavelength.txt", np.array([wavelength]), fmt = '%0.5f', delimiter = '\t')

    return(light_img, smp_img)

def load_float(smp_path):
    smp = socio.openfloat(smp_path)
    wavelength = smp.getWavelengths()
    smp_img = smp[:, :, :]
    return smp_img, wavelength

def write_img(dirname, smp, wimax, wimin):
    #フォルダを作成
    if os.path.isdir("fcimg_"+dirname) == False :
        os.mkdir("fcimg_"+dirname)
        
    #ps，npsは透過率，反射率を格納
    for i in range(smp.shape[2]):
        wavelength = 373.6399+i*(5+0.0025*i)
        socio.savefig("fcimg_"+dirname+"/smp_%04d.png" %(wavelength), smp[:,:,i], color="c", max=wimax, min=wimin)
        if i != 0 and i%50 == 0:
            print("%d" %i)
        else:
            sys.stdout.write("*")
    print("!")
        
#pick pixcel cordinate that have target intensity
def pickcord(Y,target):
    """
    ラベル付けされた領域を返す
    BGRのうち，単色にしか対応できないので注意
    基本は白黒で大丈夫
    最大３組まで
    """
    C=[]
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            if(Y[i,j]==target):
                C.append((i,j))
    return C

def ref_plt(X, Y, legends="", color="b"):
    u""" 反射率をグラフ化する関数．

    （X，Y，"凡例"，"色"）
    """
    sns.set_style("whitegrid")
    plt.plot(X, Y, "-k", linewidth=2, label=legends, color=color)

def graph_show(ylim=[0, 1], xlim=[300, 1100], ylabel="Reflectance", xlabel="Wavelength"):
    u""" グラフを表示する

    ([y軸の最小値，最大値]，[x軸の最小値，最大値]，"y軸名"，"x軸名")
    """
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(labelsize=18)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.legend(frameon=True, loc="upper center", fontsize=20).get_frame().set_edgecolor("k")
    plt.tight_layout()
    plt.show()

def graph_save(fname="test.png", ylim=[0, 1], xlim=[300, 1100], ylabel="Reflectance", xlabel="Wavelength"):
    u""" グラフを保存する（表示はしない）

    ([y軸の最小値，最大値]，[x軸の最小値，最大値]，"y軸名"，"x軸名")
    """
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(labelsize=18)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.legend(loc="upper center", fontsize=20, borderaxespad=0)
    plt.tight_layout()
    plt.savefig(fname)
