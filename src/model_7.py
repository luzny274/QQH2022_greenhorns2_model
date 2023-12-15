import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
import sklearn.preprocessing

#########################################################################################################
#########################################################################################################
#process_data.py
#########################################################################################################
#########################################################################################################

debug = False

if debug:
    import time


class Properties:
    n = 2                                                      # pocet domacich a hosticich zapasu, ktere se maji brat v potaz (v historii)
    m = 22                                                     # pocet dalsich zapasu, ktere se maji brat v potaz, tyto se ale budou podavat modelu v sume
    pocet_sloupcu = 14                                         # pocet sloupcu, ktere se ukladaji do historie tymu
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
            radekH = np.array([hsc[i], oddsA[i], oddsH[i], asc[i], h[i], a[i], sh[i], sa[i], pim_h[i], pim_a[i], ppg_h[i], ppg_a[i], fow_h[i], fow_a[i]])
            radekA = np.array([asc[i], oddsH[i], oddsA[i], hsc[i], a[i], h[i], sa[i], sh[i], pim_a[i], pim_h[i], ppg_a[i], ppg_h[i], fow_a[i], fow_h[i]])
            
            # V historii daneho tymu odstranime prvni radek a pridame dalsi, ve kterem je historie zpracovavaneho zapasu
            p.h_data_tymu[hid[i]] = np.concatenate((p.h_data_tymu[hid[i], 1:p.n + p.m, :], radekH.reshape(1, -1)), axis = 0)
            p.a_data_tymu[aid[i]] = np.concatenate((p.a_data_tymu[aid[i], 1:p.n + p.m, :], radekA.reshape(1, -1)), axis = 0)

    target = np.array(target)
    seznam_datumu = np.array(seznam_datumu)

    return [seznam_datumu, data_for_transform, target, p, sz] # Vracime data k transformaci, cile, globalni promennou udrzujici historii a velikost zpracovavanych dat

#########################################################################################################
#########################################################################################################
#retrain_whole.py
#########################################################################################################
#########################################################################################################
# To same jako "train.py", jen to model vytrenuje na vsech datech

#       Data preprocessing
global datumy
global data_for_transform
global target
global sz
datumy, data_for_transform, target, sz = np.array([], dtype=np.datetime64), np.array([]).reshape(0, p.pocet_vsech_sloupcu), np.array([]), 0

#   Feature engineering

#Nafitovani transformeru a transformace dat
# scaled_data = scaler.fit_transform(data_for_transform[:, p.num_indexy])

# data_final = scaled_data

#       Trenink

h = (10)                  # Rozlozeni vrstev k prohledani

max_iter = 2000

        #Vytvoreni a trenovani modelu
global mlp
mlp = MLPClassifier(random_state=1, hidden_layer_sizes=h , max_iter = max_iter)

global iterace
iterace = 0

global best_lambda
best_lambda = 5.4

global scores
scores = []

global last_datasize
last_datasize = 0

global mlps
mlps = []

global scalers
scalers = []

def okolni_body(prostredek, pomer, pocet_do_strany):
    zlevas = []
    zpravas = []

    posledni_pomer = pomer
    for _ in range(pocet_do_strany):
        zlevas.append( prostredek * pomer)
        zpravas.append(prostredek / pomer)

        pomer = (posledni_pomer + 1) / 2
        posledni_pomer = pomer

    whole = np.append(np.array(zlevas), prostredek)
    whole = np.append(whole, np.flip(np.array(zpravas)))
    return whole

class Model:
    

    def place_bets(self, opps, summary, inc):
        global datumy
        global iterace
        global mlp
        global data_for_transform
        global target
        global sz
        global p

        global best_lambda
        global last_datasize
        
        global mlps 
        global scalers
        global scores
        
        N = len(opps)
        
        min_bet = summary.iloc[0].to_dict()['Min_bet']
        max_bet = summary.iloc[0].to_dict()['Max_bet']
        bankroll = summary.iloc[0].to_dict()['Bankroll']
        bets = np.zeros((N, 2))
        
        number_of_mlps = 5
        minimal_score = 0.605
        minimal_datasize = 3000

        #Zpracovani inkrementalnich dat   
        if len(inc) > 0:   
            iterace += 1  
            if debug:
                start = time.time()
            new_datumy, new_data_for_transform, new_target, p, new_sz = process_data(inc) # Zpracujeme inkrementalni data, cimz aktualizujeme historie tymu (Zbytek vystupu funkce nepotrebujeme)

            if new_data_for_transform.shape[0] > 0:

                sz += new_sz
                data_for_transform = np.concatenate((data_for_transform, new_data_for_transform), axis = 0)
                target = np.concatenate((target, new_target), axis = 0)
                datumy = np.concatenate((datumy, new_datumy), axis = 0)

                pomer_datasetu = 101/100

                #Nafitovani transformeru a transformace dat
                if iterace == 1 or (pomer_datasetu * last_datasize <= data_for_transform.shape[0]):
                    # print(data_for_transform.shape[0], pomer_datasetu, last_datasize, pomer_datasetu * last_datasize, pomer_datasetu * last_datasize >= data_for_transform.shape[0])
                    last_datasize = data_for_transform.shape[0]
                    
                    scaler = sklearn.preprocessing.RobustScaler()
                    scaled_data = scaler.fit_transform(data_for_transform[:, p.num_indexy])
                    
                    data_final = scaled_data
                    # print(data_final, target)
                    # print(data_final.shape)
                    #viz process_data.py

                    # ls = np.geomspace(best_lambda * 3 / 4, best_lambda * 4 / 3, 5)
                    # ls = np.geomspace(best_lambda / 2, best_lambda * 2, 20)
                    ls = okolni_body(best_lambda, 7/8, 3)
                    f = 5
                    parameters = {'alpha' : ls}

                    if debug:
                        trans_time = time.time()-start
                        start = time.time()
                    clf = sklearn.model_selection.GridSearchCV(mlp, parameters, cv = f, n_jobs = -1, refit = True)
                    clf.fit(data_final, target)

                    best_lambda = clf.best_params_["alpha"]

                    if True:
                        mlps.append(clf.best_estimator_)
                        scores.append(clf.best_score_)
                        scalers.append(scaler)
                    if len(mlps) > number_of_mlps:
                        mlps = mlps[1:]
                        scores = scores[1:]
                        scalers = scalers[1:]

                        
                    if debug:
                        print("trans: {:.1f}s| train: {:.1f}s | scr: {:.3f}% l: {:.3f}".format(trans_time, time.time()-start, 100.0 * score, best_lambda))
                        

        if N > 0 and len(mlps) >= number_of_mlps and data_for_transform.shape[0] > minimal_datasize:
            
            #Id tymu, kteri proti sobe hrajou
            hid = opps["HID"].to_numpy()
            aid = opps["AID"].to_numpy()

            #Kurzy na tymy, kteri proti sobe hrajou
            oddsH = opps["OddsH"].to_numpy()
            oddsA = opps["OddsA"].to_numpy()
            
            # celkove sazky na zapasy z predchozich sazkarskych prilezitosti
            prev_betsH, prev_betsA = opps["BetH"].to_numpy(), opps["BetA"].to_numpy()
            
            # celkovy soucet sazek vsazenych pro dany dataframe prilezitosti
            # kontroluji, abych nepresahl svuj soucasny bankroll, v tom pripade jsou vsecny sazky ignorovany
            suma_sazek = 0
            muzes_sazet = True # pomocna promenna

            for i in range(N): # Predvidani kazdeho zapasu k sazeni
                radek_dat, zahodit = zpracuj_radek_data(hid[i], aid[i], oddsH[i], oddsA[i]) # Vytvorime data ke vstupu do modelu
                #viz process_data.py

                # Data transformujeme
                pocet = 0
                score = 0
                pred = np.array([0, 0])
                for ii in range(len(mlps)):
                    scaled_data = scalers[ii].transform(radek_dat[:, p.num_indexy])
                    data_final = scaled_data
                    cur_pred = mlps[ii].predict_proba(data_final)[0]
                    pred = pred + cur_pred
                    score = score + scores[ii]
                    # print(cur_pred)
                    pocet += 1
                # print("\n")
                pred /= pocet
                score /= pocet

                # Predvidame "pravdepodobnosti"
                pred_ind = np.argmax(pred)

                # FLAT BETTING (sazime konstantni castku)
                if False:
                    bet = max_bet / 4

                    odds = [oddsH[i], oddsA[i]]

                    if not zahodit: # Pokud nepredvidame na zaklade prazdne historie, tak sazime
                        if pred[pred_ind] * odds[pred_ind] > 1.0: # Pokud ma vyhrat dany tym s pravdepodobnosti vetsi nez 100% deleno kurzem, sazime na nej
                            bets[i, pred_ind] = bet

                # FRACTIONAL KELLY BET (bez zohledneni/se zohlednenim predchozich sazek)
                if False:
                    if not zahodit:
                        coef = 1/6 # fraction
                        pH, pA = pred[0], pred[1]
                        
                        oddH = oddsH[i] - 1
                        oddA = oddsA[i] - 1

                        EH, EA = pH*oddH - pA, pA*oddA - pH # stredni hodnota vydelku, kdyz sazim na Home/Away team, jako procentualni cast bankrollu
                        fH, fA = pH - (1-pH)/oddH, pA - (1-pA)/oddA # Kelly bet

                        tol = 0.08
                        
                        betH, betA = 0, 0
                        if EH > tol:
                            betH = coef*fH*bankroll - prev_betsH[i]    # mohu zohlednit predchozi sazky a celkove nesazet vice ne doporucuje fractional Kelly
                            if betH < min_bet: betH = 0
                            elif betH >= max_bet: betH = max_bet
                        if EA > tol:
                            betA = coef*fA*bankroll - prev_betsA[i]    # mohu zohlednit predchozi sazky a celkove nesazet vice ne doporucuje fractional Kelly
                            if betA < min_bet: betA = 0
                            elif betA >= max_bet: betA = max_bet
                        
                        if suma_sazek + betH + betA < bankroll: bets[i] = [betH, betA] # kontroluji, zda jsem nepresahl bankroll
                        else:
                            if muzes_sazet: # zmensim sazky tak, abych se vesel do bankrollu
                                x = (bankroll - suma_sazek)/(betH + betA)
                                betH, betA = x*betH, x*betA
                                muzes_sazet = False # dale uz pouze nulove sazky (presahl bych bankroll), nenulove sazky pripadne az v dalsich prilezitostech
                            else:
                                bets[i] = [0, 0]
                        suma_sazek += betH + betA

            
                # ADAPTIVE FRACTIONAL KELLY BET (bez zohledneni/se zohlednenim predchozich sazek)
                if False:
                    if not zahodit:
                        pH, pA = pred[0], pred[1]

                        oddH = oddsH[i] - 1
                        oddA = oddsA[i] - 1

                        EH, EA = pH*oddH - pA, pA*oddA - pH # stredni hodnota vydelku, kdyz sazim na Home/Away team, jako procentualni cast bankrollu
                        var = 0.289 # v puvodnim clanku 0.289, nula vychazi nejlepe (nula odpovida konstantnimu coef = 1.0)

                        betH, betA = 0, 0
                        if EH > 0:
                            fH = pH - (1-pH)/oddH # Kelly bet
                            coef = ((oddH+1)*pH-1)**2/(((oddH+1)*pH-1)**2+((oddH+1)*var)**2) # prevzato z clanku
                            betH = coef*fH*(bankroll - suma_sazek) - prev_betsH[i]    # mohu zohlednit predchozi sazky a celkove nesazet vice ne doporucuje fractional Kelly
                            if betH < min_bet: betH = 0
                            elif betH >= max_bet: betH = max_bet

                        if EA > 0:
                            fA = pA - (1-pA)/oddA # Kelly bet
                            coef = ((oddA+1)*pA-1)**2/(((oddA+1)*pA-1)**2+((oddA+1)*var)**2) # prevzato z clanku
                            betA = coef*fA*(bankroll - suma_sazek) - prev_betsA[i]    # mohu zohlednit predchozi sazky a celkove nesazet vice ne doporucuje fractional Kelly
                            if betA < min_bet: betA = 0
                            elif betA >= max_bet: betA = max_bet
                            
                        if suma_sazek + betH + betA < bankroll: bets[i] = [betH, betA] # kontroluji, zda jsem nepresahl bankroll
                        else:
                            if muzes_sazet: # zmensim sazky tak, abych se vesel do bankrollu
                                x = (bankroll - suma_sazek)/(betH + betA)
                                betH, betA = x*betH, x*betA
                                muzes_sazet = False # dale uz pouze nulove sazky (presahl bych bankroll), nenulove sazky pripadne az v dalsich prilezitostech
                            else:
                                bets[i] = [0, 0]
                        suma_sazek += betH + betA

                        
                # ADAPTIVE FRACTIONAL KELLY BET (bez zohledneni/se zohlednenim predchozich sazek)
                if True:
                    if not zahodit:

                        pH = score if pred_ind == 0 else 1.0 - score
                        pA = 1.0 - pH

                        oddH = oddsH[i] - 1
                        oddA = oddsA[i] - 1

                        EH, EA = pH*oddH - pA, pA*oddA - pH # stredni hodnota vydelku, kdyz sazim na Home/Away team, jako procentualni cast bankrollu
                        var = 0.289 # v puvodnim clanku 0.289, nula vychazi nejlepe (nula odpovida konstantnimu coef = 1.0)
                        
                        betH, betA = 0, 0
                        if EH > 0 and pred_ind == 0:
                            fH = pH - (1-pH)/oddH # Kelly bet
                            # print(fH)
                            coef = ((oddH+1)*pH-1)**2/(((oddH+1)*pH-1)**2+((oddH+1)*var)**2) # prevzato z clanku
                            betH = coef*fH*(bankroll - suma_sazek) - prev_betsH[i]    # mohu zohlednit predchozi sazky a celkove nesazet vice ne doporucuje fractional Kelly
                            if betH < min_bet: betH = 0
                            elif betH >= max_bet: betH = max_bet

                        if EA > 0 and pred_ind == 1:
                            fA = pA - (1-pA)/oddA # Kelly bet
                            # print(fA)
                            coef = ((oddA+1)*pA-1)**2/(((oddA+1)*pA-1)**2+((oddA+1)*var)**2) # prevzato z clanku
                            betA = coef*fA*(bankroll - suma_sazek) - prev_betsA[i]    # mohu zohlednit predchozi sazky a celkove nesazet vice ne doporucuje fractional Kelly
                            if betA < min_bet: betA = 0
                            elif betA >= max_bet: betA = max_bet
                            
                        if suma_sazek + betH + betA < bankroll: bets[i] = [betH, betA] # kontroluji, zda jsem nepresahl bankroll
                        else:
                            if muzes_sazet: # zmensim sazky tak, abych se vesel do bankrollu
                                x = (bankroll - suma_sazek)/(betH + betA)
                                betH, betA = x*betH, x*betA
                                muzes_sazet = False # dale uz pouze nulove sazky (presahl bych bankroll), nenulove sazky pripadne az v dalsich prilezitostech
                            else:
                                bets[i] = [0, 0]
                        suma_sazek += betH + betA


        
        return pd.DataFrame(data=bets, columns=['BetH', 'BetA'], index=opps.index) # Vracime sazky
