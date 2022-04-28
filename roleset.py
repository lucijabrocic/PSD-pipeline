from spacy.language import Language
from spacy.tokens import Token, Doc, Span
import os
import joblib

from struct2 import IOBType, LabelSpan
from srl import *




from spacy.language import Language
from spacy.tokens import Token, Doc, Span
import os
import joblib

import allennlp_models.pretrained
srlbert=allennlp_models.pretrained.load_predictor("structured-prediction-srl-bert")


class RolesetComponent(object):
    name = "roleset"

    @staticmethod
    def doc_roleset_getter(doc):
        doc_roleset = []
        for token in doc:
            doc_roleset.append(token._.roleset)
        return doc_roleset

    @staticmethod
    def span_roleset_getter(span):
        span_roleset = []
        for token in span:
            span_roleset.append(token._.roleset)
        return span_roleset


    def _extract_features(self, token_data, pos_flag=True, dep_flag=True):

        def _lemmatize_ing(token, vocab):
            verb_ing=token.text
            # initial candidates from vocab which share same first letter
            candidates = [verb_inf
                          for verb_inf in vocab 
                          if verb_ing and verb_ing[0].lower() == verb_inf[0].lower()]

            if token.lemma_ in candidates: 
                return token.lemma_
            last_letters=len(set(verb_ing[:-3][-2:])) 
            # filter out candidates by next letter (greedy)
            for i in range(1, len(verb_ing) - 5 + last_letters):
                next_candidates = []
                for verb_inf in candidates:
                    if i < len(verb_inf) and verb_ing[i].lower() == verb_inf[i].lower():
                        next_candidates.append(verb_inf)
                if not next_candidates:
                    break
                candidates = next_candidates
            
            
            verb_inf = ""
            if len(candidates) == 1:  # only one candidate
                verb_inf = candidates[0]       
            elif len(candidates) > 1:  # more candidates, take one whose length is closest to verb with ing
                verb_inf = min(candidates, 
                               key=lambda verb_inf: abs(len(verb_ing) - 5 + last_letters - len(verb_inf)))
            if not verb_inf:
                verb_inf = verb_ing[:-3]  # cut -ing

            # calculate ratio of similarity
            n = min(len(verb_inf), len(verb_ing) - 3)
            total = n * (n + 1) // 2
            first = verb_ing[:n].lower()
            second = verb_inf[:n].lower()
            cnt = sum([int(first[i] == second[i]) * (i + 1) for i in range(n)])
            ratio = cnt / total
            if ratio >= 0.9:
                return verb_inf
            return verb_ing[:-3]


        def _lemmatize_ap(token): #lemmatize_apostrophe
            lemma=token.lemma_
            if lemma in ["'m", "'re", "'s"]:
                return "be"
            if lemma =="'ve":
                return "have"
            if lemma in ["'ll"]:
                return "will"
            if lemma in ["n't"]:
                return "not"
            if lemma=="'d":
                if token.head.tag_=="VBN" or token.nbor().text in ["better", "best"]:
                    return "have"
                elif token.head.tag_=="VB":
                    return "would"

        token=token_data[0]
        lemma=""
        text = token.text
        pos = token.tag_
        dep = token.dep_
        srl=token_data[1]
        features_dict={
              "text": token.text, # token text
              "lemma": lemma, # token lemma, if we don't predict roleset for this token, the value is "-"
          }
        labels=["ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5", "ARG6", "ARGM-COM", "ARGM-LOC", "ARGM-DIR","ARGM-GOL", "ARGM-MNR","ARGM-TMP", "ARGM-EXT", "ARGM-REC", "ARGM-PRD", "ARGM-PRP", "ARGM-CAU", "ARGM-DIS",
        "ARGM-ADV", "ARGM-ADJ", "ARGM-MOD", "ARGM-NEG", "ARGM-DSP", "ARGM-LVB", "ARGM-CXN", "ARGM-PRR"]
        if srl and token.tag_!="MD":

            lemma=self.lemma_frameset.get(token.lemma_, token.lemma_)
            if lemma.endswith("ing"):# and not token.tag_.startswith("NN"):
                lemma=_lemmatize_ing(token, self.lemma_frameset)
            if "'" in lemma:
                lemma=_lemmatize_ap(token)

            features_dict["lemma"]=lemma
            for label in labels:
                if pos_flag and dep_flag:
                    features_dict[label] = [pos+"_"+dep for span in srl if span.label_ in [label, "C-"+label, "R-"+label] for t in span] 
                elif pos_flag:
                    features_dict[label] = [pos for span in srl if span.label_ in [label, "C-"+label, "R-"+label] for t in span] 
                elif dep_flag:
                    features_dict[label] = [dep for span in srl if span.label_ in [label, "C-"+label, "R-"+label] for t in span] 

            # for i, label in enumerate(srl.split(";")):
            #     if label!="O":
            #         label.replace("C-", "").replace("R-", "")
            #         if pos_flag and dep_flag:
            #             features_dict[label] = features_dict.get(label, [])+[tagged_sentence[i][1]+"_"+tagged_sentence[i][2]]
            #         elif pos_flag:
            #             features_dict[label] = features_dict.get(label, [])+[tagged_sentence[i][1]]
            #         elif dep_flag:
            #             features_dict[label] = features_dict.get(label, [])+[tagged_sentence[i][2]]
                    






        # for label in labels:
        #     features_dict[label]=[t.tag_ for span in srl if span.label_ in [label, "C-"+label] for t in span]

        for label in labels:
            if label not in features_dict:
                features_dict[label] = []

        return features_dict


    def _vectorize(self, X_features, vocab):
        result=[]
        for i, token in enumerate(X_features):
            for lbl, val in token.items(): 
                # if isinstance(v, str):
                if lbl in ["text", "lemma"]:
                    feat=f'{lbl}={val}'
                    if feat in vocab:
                        result.append((i, vocab[feat]))
                else:
                    for pos in val:
                        feat=f'{lbl}={pos}'
                        if feat in vocab:
                            result.append((i, vocab[feat]))

        return result


    def _classify(self, x, len_vocab, len_sent, verb):
        intercept=self.predictors[verb]["class_intercept"]
        coef=[(pos//len_vocab, pos%len_vocab, val) for (pos, val) in self.predictors[verb]["class_coef"]]
        result=[[]]*len_sent

        for i in range(len_sent):
            temp = {}
            for xi, xj in x:
                if xi == i:
                    temp[xj] = temp.get(xj, 0) + 1
            
            #temp=[xj for xi,xj in x if xi==i] # di se nalaze jedinice
            result[i]=[sum([cval*temp.get(cj, 0) for ci, cj, cval in coef if ci==j]) + intercept[j] 
                       for j in range(len(intercept))]
        return result

    def _find_frame_for_lemma(self, lemma):
        frame=self.lemma_frameset.get(lemma)
        if frame: 
            return frame

        for f, aliases in self.frameset_aliases.items():
            if lemma in [alias for pos, alias_role in aliases.items() 
                                      for alias, role in alias_role]:
                return f
        else:
            #raise Exception(f"No '{lemma}' in lemma_frameset!")
            print(f"No '{lemma}' in lemma_frameset!")
            return None


    def _get_roleset_for_alias_or_lemma(self, alias_or_lemma, frame, pos):
        t=[roleset for alias_name, roleset in self.frameset_aliases[frame][pos] if alias_name==alias_or_lemma]
        if len(t)==1: # može se jedinstveno odredit
            return t[0]
        elif len(t)>1: # ne može jedinstveno
            return min(t, key=lambda x: x.split(".")[-1])+"X" 
        else: 
            pass # ne postoji


    def _fix_prediction_lvb(self, sent_info, lemmas, prediction):
        fixed_prediction=[]
        for token in sent_info:
            i=token.i - token.sent.start
            srl=[(span.label_, tok) for span in sent_info[token] for tok in span]
            is_srl_verb=srl and token.tag_!="MD" # da uzme samo glagole bez modalnih
            token_prediction=None
            has_prediction=bool(token_prediction)
            
            if is_srl_verb and not(has_prediction): # ako je glagol, ali nema predikciju

                frame=self._find_frame_for_lemma(lemmas[i])
                lvb=[tok for lbl, tok in srl if "ARGM-LVB" == lbl]

                if lvb:
                    lvb=lvb[0]
                    #lvc=f"{lvb.lemma_}_{token.lemma_}"
                    lvb_i=lvb.i - lvb.sent.start
                    lvc=f"{lemmas[lvb_i]}_{lemmas[i]}"
                    
                    ### PROMJENJEN DIO S FRAME-OM
                    if frame and "l" in self.frameset_aliases[frame]: # ako postoji oznaka (l)
                        token_prediction=self._get_roleset_for_alias_or_lemma([lvc, token.lemma_], frame, "l")

                    if frame and "n" in self.frameset_aliases[frame] and not token_prediction: # ako nema (l), treba provjeriti (n) oznaku
                        token_prediction=self._get_roleset_for_alias_or_lemma([token.lemma_, lemmas[i]], frame, "n")

                    if not frame or not token_prediction: # ne postoji ni l ni n (kao npr money)
                        token_prediction=f"{lemmas[i]}.00"
                else:
                    token_prediction=prediction[i]

                
            fixed_prediction.append(token_prediction)
                
        return fixed_prediction


    def _fix_prediction_verbs(self, sent_info, lemmas, prediction):
        fixed_prediction=[]
        for token in sent_info:
            i=token.i - token.sent.start
            srl=[(span.label_, tok) for span in sent_info[token] for tok in span]
            is_srl_verb=srl and token.tag_!="MD" # da uzme samo glagole bez modalnih
            token_prediction=prediction[i]
            has_prediction=bool(token_prediction)
            
            if is_srl_verb and not(has_prediction): # ako je glagol, ali nema predikciju
                frame=self._find_frame_for_lemma(lemmas[i]) # treba li ova linija?? s obzirom da je u lemmas spremljen frameset -- treba zbog problema kao "am", "president" i sl.
                if frame and "v" in self.frameset_aliases[frame]: ### PROMJENA
                    token_prediction=self._get_roleset_for_alias_or_lemma([token.lemma_, lemmas[i]], frame, "v")
                    
            fixed_prediction.append(token_prediction)
        return fixed_prediction


    def _fix_prediction_phrasal(self, sent_info, lemmas, prediction):
        fixed_prediction=[]
        for token in sent_info:
            i=token.i - token.sent.start
            srl=[(span.label_, tok) for span in sent_info[token] for tok in span]
            is_srl_verb=srl and token.tag_!="MD" # da uzme samo glagole bez modalnih
            token_prediction=prediction[i]
            if is_srl_verb:
                for child in token.children:
                    if child.dep_=="prt":
                        temp=lemmas[i]+"_"+child.lemma_
                        frame=self._find_frame_for_lemma(lemmas[i])
                        if not frame:
                                frame=self._find_frame_for_lemma(temp)
                        if frame and "v" in frameset_aliases[frame]:
                            token_prediction=self._get_roleset_for_alias_or_lemma([temp, token.lemma_+"_"+child.lemma_], frame, "v")
                        break
            fixed_prediction.append(token_prediction)
        return fixed_prediction





                    
    def _get_roleset_for_alias_or_lemma(self, alias_or_lemma, frame, pos):
        # t=[roleset for alias_name, roleset in self.frameset_aliases[frame][pos] if alias_name==alias_or_lemma]
        t=[roleset for alias_name, roleset in self.frameset_aliases[frame][pos] if alias_name==alias_or_lemma[0]]

        if len(t)==0:
            t=[roleset for alias_name, roleset in self.frameset_aliases[frame][pos] if alias_name==alias_or_lemma[1]]
        if len(t)==1: # može se jedinstveno odredit
            return t[0]
        elif len(t)>1: # ne može jedinstveno
            return min(t, key=lambda x: x.split(".")[-1])+"X" 
        else: 
            pass # ne postoji
                    


    def _predict_model(self, sent_info):
        X_features=[self._extract_features(item) for item in sent_info.items()]
        framesets_sent={t["lemma"] for t in X_features}-{""}
    
        pred=[None]*len(sent_info)
        for frame in framesets_sent:
            if frame not in self.predictors:
                continue
            vocab=self.predictors[frame]["vec_vocab"]
            coef=self.predictors[frame]["class_coef"]
            intercept=self.predictors[frame]["class_intercept"]
            classes=self.predictors[frame]["class_classes"]


            x=self._vectorize(X_features, vocab)
            y=self._classify(x, len(vocab), len(sent_info), frame)

            if len(classes)>2:
                for i, y_row in enumerate(y):
                    if X_features[i]["lemma"]==frame:
                        m=max(range(len(y_row)), key=y_row.__getitem__)
                        pred[i]=classes[m] if m>0 else None
            else:
                y=[y_el for y_row in y for y_el in y_row]
                for i, y_el in enumerate(y):
                    if y_el>0 and X_features[i]["lemma"]==frame: pred[i]=classes[1]
               
        return pred

    
    def __init__(self, nlp: Language, model=""):

        import json
        print("Model:", model)
        with open(model, "r") as json_file:
            temp=json.load(json_file)
            self.rolesets=temp["roleset"]
            self.predictors=temp["predictors"]
        
        import numpy as np
        for verb in self.predictors:
            self.predictors[verb]["class_coef"]=eval(self.predictors[verb]["class_coef"])
            self.predictors[verb]["class_intercept"]=np.array(eval(self.predictors[verb]["class_intercept"]))
            self.predictors[verb]["class_classes"]=np.array(eval(self.predictors[verb]["class_classes"]))

        self.labels=tuple(self.rolesets.keys())
        self.lemma_frameset={r["lemma"]:r["frameset"] for r in self.rolesets.values()}

        self.frameset_aliases={}

        for roleset, val in self.rolesets.items():
            for lemma, pos in val["aliases"]:
                self.frameset_aliases[val["frameset"]]=self.frameset_aliases.get(val["frameset"], {})
                self.frameset_aliases[val["frameset"]][pos]=self.frameset_aliases[val["frameset"]].get(pos, [])+[(lemma, roleset)]

        self.register()

    @staticmethod
    def register():
        # Register custom extension on the Token, Span, Doc
        Token.set_extension("roleset", default=None, force=True)
        Span.set_extension("roleset", getter=RolesetComponent.span_roleset_getter, force=True)
        Doc.set_extension("roleset", getter=RolesetComponent.doc_roleset_getter, force=True)



    def __call__(self, doc: Doc) -> Doc:
        for sent in doc.sents:
            # sent_info --> sent._.srl + prazne liste
            sent_info={}
            for token in sent:
                if token in sent._.srl: 
                    sent_info[token]=sent._.srl[token]
                else: 
                    sent_info[token]=[] 


            X_features=[self._extract_features(item) for item in sent_info.items()]
            lemmas=[tok["lemma"] for tok in X_features]
            prediction1=self._predict_model(sent_info)
            prediction2=self._fix_prediction_lvb(sent_info, lemmas, prediction1)
            prediction3=self._fix_prediction_verbs(sent_info, lemmas, prediction2)
            prediction=self._fix_prediction_phrasal(sent_info, lemmas, prediction3)

            for i, t in enumerate(sent):
                t._.roleset=prediction[i]
             
        return doc

    

def add_roleset_to_pipe(nlp):
    
    @Language.factory("roleset_component", default_config={"model": "./psd_model_pos_dep.json"}, assigns=["doc._.roleset", "span._.roleset", "token._.roleset"])
    def create_roleset_component(nlp, name, model):
        return RolesetComponent(nlp, model)

    if "roleset" in nlp.pipe_names:
        nlp.replace_pipe(factory_name="roleset_component", last=True)
    else:
        nlp.add_pipe(factory_name="roleset_component", last=True)

    return nlp
