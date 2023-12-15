# Pomocny skript vyuzivany pri treninku i evaluation pro zpracovani dat

import pandas as pd
import numpy as np
import sklearn.preprocessing
import sklearn.compose

class Properties:
    n = 2                                                      # pocet domacich a hosticich zapasu, ktere se maji brat v potaz (v historii)
    m = 22                                                     # pocet dalsich zapasu, ktere se maji brat v potaz, tyto se ale budou podavat modelu v sume
    pocet_sloupcu = 22                                         # pocet sloupcu, ktere se ukladaji do historie tymu
    pocet_tymu = 1                                             # pocet tymu, postupne se zvysuje, kdyz se narazi na nove tymy
    h_data_tymu = np.zeros((pocet_tymu, n + m, pocet_sloupcu)) # historie domacich zapasu jednotlivych tymu
    a_data_tymu = np.zeros((pocet_tymu, n + m, pocet_sloupcu)) # historie hosticich zapasu jednotlivych tymu

    pocet_vsech_sloupcu = pocet_sloupcu * (n + 1) * 2 + 2      # pocet vsech sloupcu v radku na vstupu modelu (pred transformaci)

    num_indexy = range(0, pocet_vsech_sloupcu)                 # indexy pro skalovani

    def __init__(self):
        self.h_data_tymu[:, :, 0] = -1            # oznaceni prazdne historie pomoci -1, na pozici s hodnotami pouze kladnymi
        self.a_data_tymu[:, :, 0] = -1

    def rozsir_data_tymu(self, novy_tym_id):    # kdyz narazime na id noveho tymu, pak je treba rozsirit matice historie tymu
        new_pocet_tymu = novy_tym_id + 1

        new_h_data_tymu = np.zeros((new_pocet_tymu, self.n + self.m, self.pocet_sloupcu))
        new_h_data_tymu[:self.pocet_tymu] = self.h_data_tymu
        new_h_data_tymu[self.pocet_tymu:, :, 0] = -1

        new_a_data_tymu = np.zeros((new_pocet_tymu, self.n + self.m, self.pocet_sloupcu))
        new_a_data_tymu[:self.pocet_tymu] = self.a_data_tymu
        new_a_data_tymu[self.pocet_tymu:, :, 0] = -1

        self.pocet_tymu = new_pocet_tymu
        self.h_data_tymu = new_h_data_tymu
        self.a_data_tymu = new_a_data_tymu

global p
p = Properties()    # Globalni promenna, ktera v sobe udrzuje historii tymu a zakladni informace

def zpracuj_radek_data(hid, aid, oddsH, oddsA):  # Vytvoreni jednoho radku dat pro vstup do modelu (jeste bez transformace)
    global p
    
    if max(hid, aid) >= p.pocet_tymu:    # kdyz narazime na id noveho tymu, pak je treba rozsirit matice historie tymu
        p.rozsir_data_tymu(novy_tym_id = max(hid, aid))

    exaktni_historie_h = p.h_data_tymu[hid][p.m:p.m + p.n] # Exaktni historie nejnovejsich n zapasu
    exaktni_historie_a = p.a_data_tymu[aid][p.m:p.m + p.n]
    suma_historie_h = np.sum(p.h_data_tymu[hid][:p.m], axis = 0).reshape(1, -1) # Suma historie dalsich m zapasu
    suma_historie_a = np.sum(p.a_data_tymu[aid][:p.m], axis = 0).reshape(1, -1)

    radek_dat = np.concatenate((exaktni_historie_h, exaktni_historie_a, suma_historie_h, suma_historie_a), axis = 0)
    
    radek_dat = radek_dat.reshape(1, -1) # Matici prevedeme na radkovy vektor                                        
    radek_dat = np.concatenate((radek_dat, np.array([oddsH, oddsA]).reshape(1, -1)), axis = 1) # Bereme v uvahu soucasny kurz

    zahodit = (p.h_data_tymu[hid, 0, 0] == -1) or (p.a_data_tymu[aid, 0, 0] == -1) # -1 znaci prazdnou historii, pokud zjistime, ze se o ni jedna, tak tento radek zahodime, abychom model zbytecne nematli

    return radek_dat, zahodit # Vracime radek a zda ho zahodit

def process_data(data):
    global p

    # Rozdelime si panda dataframe na sloupecky a prevedeme na numpy array
    
    datumy = data["Date"].to_numpy() # Unikátní identifikátor domácího týmu

    hid = data["HID"].to_numpy() # Unikátní identifikátor domácího týmu
    aid = data["AID"].to_numpy() # Unikátní identifikátor týmu hostí

    oddsH = data["OddsH"].to_numpy() # Bookmakerovy kurzy pro daný výsledek
    oddsA = data["OddsA"].to_numpy()

    hsc = data["HSC"].to_numpy() # Výsledné skóre domácích
    asc = data["ASC"].to_numpy() # Výsledné skóre hostí

    h = data["H"].to_numpy() # Binární indikátor výhry domácích/hostí
    a = data["A"].to_numpy()

    sh = data["S_H"].to_numpy() # Střely na bránu daného týmu
    sa = data["S_A"].to_numpy()

    pim_h = data["PIM_H"].to_numpy() # Trestné minuty daného týmu
    pim_a = data["PIM_A"].to_numpy()

    ppg_h = data["PPG_H"].to_numpy() # Góly v přesilovkách daného týmu
    ppg_a = data["PPG_A"].to_numpy()

    fow_h = data["FOW_H"].to_numpy() # Vyhraná vhazování daného týmu
    fow_a = data["FOW_A"].to_numpy()

    giv_h = data["GIV_H"].to_numpy() # GIV_H/A - Ztráty puku daného týmu
    giv_a = data["GIV_A"].to_numpy()

    tak_h = data["TAK_H"].to_numpy() #  TAK_H/A - Zisky puku daného týmu
    tak_a = data["TAK_A"].to_numpy()

    hit_h = data["HIT_H"].to_numpy()# HIT_H/A - Hity daného týmu
    hit_a = data["HIT_A"].to_numpy()

    blk_h = data["BLK_H"].to_numpy()# BLK_H/A - Bloky střel daného týmu
    blk_a = data["BLK_A"].to_numpy()

    target = list() # Budouci seznam targetu
    data_for_transform = np.array([], dtype=np.float64).reshape(0, p.pocet_vsech_sloupcu) # Budouci matice dat

    sz = h.shape[0] # Pocet zpracovanych dat
    seznam_datumu = list()
    
    for i in range(sz): # Zpracovavame jeden radek dat
        #       Upravujeme data a cile pro trenovani

        if h[i] != a[i]:
            radek_dat, zahodit = zpracuj_radek_data(hid[i], aid[i], oddsH[i], oddsA[i]) #Zpracujeme radek na zaklade historii tymu a kurzu

            if not zahodit: # Pokud nemame nekde na radku prazdnou historii, tak radek pridame k datum na trenovani
                data_for_transform = np.concatenate((data_for_transform, radek_dat), axis = 0)
                seznam_datumu.append(np.datetime64(datumy[i]))

                if h[i]:            # Hodnotou 0 oznacime vyhru domacich   
                    target.append(0)
                elif a[i]:          # Hodnotou 1 oznacime vyhru hosticich   
                    target.append(1)
            
            #       Upravujeme historie tymu

            # Doplnujeme nasledujici sloupce: Zda je tym domaci (0/1), kurzy pro, kurzy proti, skore tymu, vyhra, prohra, strely pro, strely proti, trestne minuty pro, trestne minuty proti, goly v presilovkach pro, goly v presilovkach proti, vyhrana vhazovani pro, vyhrana vhazovani proti
            radekH = np.array([hsc[i], oddsA[i], oddsH[i], asc[i], h[i], a[i], sh[i], sa[i], pim_h[i], pim_a[i], ppg_h[i], ppg_a[i], fow_h[i], fow_a[i], giv_h[i], giv_a[i], tak_h[i], tak_a[i], hit_h[i], hit_a[i], blk_h[i], blk_a[i]])
            radekA = np.array([asc[i], oddsH[i], oddsA[i], hsc[i], a[i], h[i], sa[i], sh[i], pim_a[i], pim_h[i], ppg_a[i], ppg_h[i], fow_a[i], fow_h[i], giv_a[i], giv_h[i], tak_a[i], tak_h[i], hit_a[i], hit_h[i], blk_a[i], blk_h[i]])
            
            # V historii daneho tymu odstranime prvni radek a pridame dalsi, ve kterem je historie zpracovavaneho zapasu
            p.h_data_tymu[hid[i]] = np.concatenate((p.h_data_tymu[hid[i], 1:p.n + p.m, :], radekH.reshape(1, -1)), axis = 0)
            p.a_data_tymu[aid[i]] = np.concatenate((p.a_data_tymu[aid[i], 1:p.n + p.m, :], radekA.reshape(1, -1)), axis = 0)

    target = np.array(target)
    seznam_datumu = np.array(seznam_datumu)

    return [seznam_datumu, data_for_transform, target, p, sz] # Vracime data k transformaci, cile, globalni promennou udrzujici historii a velikost zpracovavanych dat
