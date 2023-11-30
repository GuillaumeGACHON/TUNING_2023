def dicoteur(liste_met,liste_sc):
    dico={}
    liste_reste=liste_met.copy()
    for el in liste_met:
        if el[-3:]=='ANM' or el[-4:]=="bias" or el[:6]=="Qt_OCE":
            dico.update({el:el})
            liste_reste.remove(el)
        elif el[-3:]=='JAS' or el[-3:]=='JFM':
            for elem in liste_sc:
                if elem[:-3]==el[:-3]:
                    dico.update({el:elem})
                    liste_reste.remove(el)
        elif el[-8:]=="seacycle":
            dico.update({el:el[:-8]+"bias"})
            liste_reste.remove(el)
        elif el=="ENSO_amplitude" or el=="ENSO_seasonality":
            dico.update({el:"ENSO_amplitude"})
            liste_reste.remove(el)
    return dico,liste_reste
