# -*- coding: utf-8 -*-
import itertools
import re, os
from interval import Interval, IntervalSet
import sqlite3
from interruptingcow import timeout

def findall(sub, string):
    index = 0 - len(sub)
    try:
        while True:
            index = string.index(sub, index + len(sub))
            yield index
    except ValueError:
        pass

def minfind(s,f):
    r=[]
    for l in f:
        u = s.find(l)
        if u >= 0:
            r.append(u)
    if r==[]: return 0
    else: return min(r)

def chercher_balise(source, type="", cond_attr={}, cond_style={}, sous_balises=[], inclure_balises=False, interval_set=False):
    """
        Arguments:
            type: type de balise (e.g. div, h1 ...)
            cond_attr: condition sur les attributs de la balise (e.g. class="c1")
            cond_style: condition sur le style css (e.g. font-style: italic)
            sous_balises: liste de types de sous-balises à inclure dans le contenu, ["all"] pour inclure tous les types de sous-balises
            inclure_balises: inclure ou non les balises d'ouverture/de fermeture dans le contenu
        Sortie:
            liste de (début, fin) correspondant aux indices délimitant le contenu de la balise recherchée
    """
    result = []
    exclude = []
    if type != "":
        starts = list(findall("<" + type + " ", source)) + list(findall("<" + type + ">", source))
    else:
        starts = list(findall("<", source))
    starts.sort()
    for start in starts:
        if source[start+1:start+2] == "/":
            continue
        if inclure_balises:
            debut_contenu = start
        else:
            debut_contenu = start + source[start:].find(">") + 1
        if type=="":
            type_ = source[start+1:start + 1 + minfind(source[start+1:],[" ",">","/"])]#Dans le cas où type="", permet de récupérer le type de balise
        else:
            type_ = type
        fin_contenu = start + source[start + 1:].find("</" + type_) + 1
        if inclure_balises:
            fin_contenu += source[fin_contenu:].find(">") + 1
        contenu = source[debut_contenu:fin_contenu]
        attributs_ = source[start+1+len(type_):start + source[start:].find(">")]

        attributs_ = attributs_.split('" ')
        attributs = {}

        for a in attributs_:
            key = a[:a.find("=")].replace(" ","")
            val = a[a.find("=")+2:].replace('"','')
            if key in attributs.keys():
                attributs[key] += val
            else:
                attributs[key] = val
        style = {}
        if 'style' in attributs.keys():
            style_ = attributs['style']
            style_ = style_.split(';')
            for s in style_:
                if s.replace(" ","") != "":
                    key = s[:s.find(":")].replace(" ", "")
                    val = s[s.find(":") + 1:].replace(" ", "")
                    style[key] = val
        cond = True
        for k, v in cond_attr.iteritems():
            if k not in attributs.keys() or attributs[k] != v:
                cond=False
        for k, v in cond_style.iteritems():
            if k not in style.keys() or style[k] != v:
                cond=False
        if cond:
            result.append((debut_contenu,fin_contenu))
            if not inclure_balises:
                for (debut_sous,fin_sous) in chercher_balise(contenu, inclure_balises=True):
                    type_sous = contenu[debut_sous+1:debut_sous+1+minfind(contenu[debut_sous+1:],[" ",">","/"])]
                    if type_sous not in sous_balises and "all" not in sous_balises:
                        exclude.append((debut_contenu+debut_sous,debut_contenu+fin_sous))
    r1 = IntervalSet([Interval(d, f) for (d,f) in result])
    r2 = IntervalSet([Interval(d, f) for (d,f) in exclude])
    R = r1 - r2
    if interval_set:
        return R
    if inclure_balises:
        return result
    else:
        return [(x.lower_bound, x.upper_bound) for x in R.intervals]

def supprimer_balises(s):
    bal = findall("<",s)
    decal = 0
    for b in bal:
        supp = s[b+decal:].find(">")+1
        s = s[:b+decal] +" "+s[b+decal+supp:]
        decal -= supp-1
    while s.replace("  "," ") != s:
        s = s.replace("  "," ")
    return s

def reecrire_styles(html):
    style = chercher_balise(html,"style")
    d,f = style[0]
    style = html[d:f].split("}")
    classes={}
    for s in style:
        if s.find("{") != -1:
            sp = s.split("{")
            cl = sp[0].replace("\n","").replace(" ","").split(",")[0]
            if cl[0] == ".":
                classes[cl.replace(".","")] = sp[1]
    bal = findall("class=",html)
    decal=0
    for d in bal:
        e = decal+d+7+html[decal+d+7:decal+d+20].find('"')
        class_ = html[decal+d+7:e]
        insert=' style="'+classes[class_].replace(' ','').replace('"','')+'"'
        decal+=len(insert)
        html = html[:e+1] + insert + html[e+1:]
    return html



def traiter_doc(docid):
    print("Traitement de "+docid)
    if os.path.exists('../data/sql/'+docid+'.db'):
        print("---Doc déjà traité")
        return
    html = open("../../data/html/" + docid + ".html", "r").read()
    print("Réécriture des styles dans les balises...")
    html = reecrire_styles(html)


    print("Recherche des titres...")
    H1 = chercher_balise(html, "h1", cond_attr={}, sous_balises=["a"])
    H1 += chercher_balise(html, "b", cond_attr={}, sous_balises=[])
    H1 = sorted(H1, key=lambda x: x[0])
    db = sqlite3.connect('../../data/sql/'+docid+'.db')
    db.text_factory = str
    cursor = db.cursor()
    #cursor.execute('''DROP TABLE documents''')
    db.commit()
    cursor.execute('CREATE TABLE documents (id varchar , text varchar)')
    db.commit()

    def process_titre(debut_titre, fin_titre, i):

        titre = html[debut_titre:fin_titre].replace(":", "").lower()
        titre = supprimer_balises(titre)
        if i == len(H1) - 1:
            debut_titre_suivant = len(html)
        else:
            debut_titre_suivant, _ = H1[i + 1]

        contenu_indices = chercher_balise(html[fin_titre:debut_titre_suivant], "a", inclure_balises=["all"],
                                          interval_set=True)
        contenu_indices += chercher_balise(html[fin_titre:debut_titre_suivant], "p", inclure_balises=["all"],
                                           interval_set=True)
        contenu_indices += chercher_balise(html[fin_titre:debut_titre_suivant], "span", cond_attr={"class": "p"},
                                           inclure_balises=["all"], interval_set=True)

        contenu_indices -= chercher_balise(html[fin_titre:debut_titre_suivant], cond_style={"font-size": "10pt"},
                                           inclure_balises=["all"], interval_set=True)
        contenu_indices -= chercher_balise(html[fin_titre:debut_titre_suivant], cond_style={"font-size": "11pt"},
                                           inclure_balises=["all"], interval_set=True)
        contenu_indices -= chercher_balise(html[fin_titre:debut_titre_suivant], cond_style={"font-size": "9pt"},
                                           inclure_balises=["all"], interval_set=True)
        contenu_indices -= chercher_balise(html[fin_titre:debut_titre_suivant], cond_style={"font-size": "8pt"},
                                           inclure_balises=["all"], interval_set=True)
        contenu_indices -= chercher_balise(html[fin_titre:debut_titre_suivant], cond_style={"font-size": "7pt"},
                                           inclure_balises=["all"], interval_set=True)

        contenu_indices = [(x.lower_bound, x.upper_bound) for x in contenu_indices.intervals]
        contenu = ""
        for (d, f) in contenu_indices:
            contenu += html[fin_titre:debut_titre_suivant][d:f] + " "
        contenu = supprimer_balises(contenu)
        regex = '(?P<lettre>([a-zA-Z)"]{2}))\.\d+'
        contenu = re.sub(regex, '\g<lettre>.', contenu)
        regex = '(?P<lettre>[a-zA-Z),;"])\d+ '
        contenu = re.sub(regex, '\g<lettre> ', contenu)
        if contenu.replace(" ", "") != "":
            cursor.execute('''INSERT INTO documents(id, text)
                              VALUES(?,?)''', (titre, contenu))
            db.commit()

            print("TITRE: " + titre)
            ##print("CONTENU: "+contenu)
            #print("-----------------------------------------------------------------------------------------------------------------------")

    i = 0
    for (debut_titre,fin_titre) in H1:
        try:
            with timeout(60 * 5, exception=RuntimeError):
                process_titre(debut_titre, fin_titre, i)
                i += 1
        except RuntimeError:
            print("Temps dépassé")
            continue
        print(
        "-----------------------------------------------------------------------------------------------------------------------")

    db.close()


for element in os.listdir('../../data/html/'):
    if element.endswith('.html') and element[0] != ".":
        traiter_doc(element.replace(".html",""))


#traiter_doc("test")
