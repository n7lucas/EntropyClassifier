import numpy as np
import random as randomizer
from graphviz import Source
from graphviz import render
from anytree.exporter import DotExporter
from anytree import NodeMixin, RenderTree
from anytree import RenderTree,AnyNode, AsciiStyle, PreOrderIter,LevelOrderIter
from anytree import Node as NOD
import seaborn as sns
#from sklearn.metrics import confusion_matrix
import math
from random import random
from anytree.exporter import UniqueDotExporter
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import streamlit as st

class EntropyClassifier ():
    def __init__(self, optimizer='Monte_Carlo', nb_iter = 10000, nb_entropy=10**-10 , nb_leaf = 15, nb_branch_max=5, activation='sigmoide' ):
        self.activation = activation
        self.optimizer = optimizer
        self.nb_iter = nb_iter
        self.nb_leaf = nb_leaf
        self.nb_branch_max = nb_branch_max
        self.nb_entropy = nb_entropy
        self.root = None
        self.total_branches = 0
 
    def _random_a(self,X):
        """
        Just declared function of generate vector a, in each Straight Line version (P1, P2,...) i will override this function
        with the suitable version
        
        Receive: X (data train input)
        
        Return: Vector a based on lenght of X, P1 (a in R^n+1), P2 (a in R^3n),...
        
        """
        
        pass
       
    def _split(self,arr, cond):
        """
        Function that will split the dataset based on condition of event be in each side of demiplane
        """
        return [arr[cond], arr[~cond]]

        
    def _sigmoid(self,s):  
        """
        Sigmoide Function, my version (not numpy vetrsion) because i want to change the value of PSI every time
        
        Input: s -> result of straightline equation, EPSI -> Value of epsi of sigmoide function
        
        Output: transformed value of straighline equation
        """
        large=30
        if s<-large: s=-large
        if s>large: s=large
        return (1 / (1 + np.exp(-s/(1/5))))
     
    def _norm_vector (self,a):
        new_a = a.copy()
        normalized_v =  new_a / np.sqrt(np.sum( new_a[0,1:]**2))
        return  normalized_v
  
    """I gona call this function inside de class of classifier"""
    "Call all function that e«generate the line"
    def _Line(self,X,a):
        """
        No code on here, i will override in each straightLine class, with the suitable version
        
        Input: X -> input data train, y -> input data test, a -> vector a
        
        Output: D_classe -> dict of set of classes, f_pos -> positive frequency, f_neg -> negative frequency,
                            D_pos -> set of events in positive demiplan, D_neg -> set of events in negative demiplan,
                            S_a_pos -> Entropy of positive demiplan, S_a_neg -> Entropy of negative demiplan
                            S_a_Total -> Total Entropy
        """
        pass
    
    def _groups_heavi(self,X):
        """
        Generate Group's of class for Heaviside Activation function, (its the same with the size of vecotr, but the 2 method 
                                                        are almost equal so use the same method)
        
    
        Input: X -> input Data train, y -> input Data test.
        """
        D_classe = {value: X[X[:,-1] == value]for value in np.unique(X[:,-1])} #
        return D_classe
    
    def _groups_sig(self,X_sig):
        """
        Generate Group's of class for Sigmoide Activation function
        
        Input: sig_func -> input Data train with class and respective sigmoide value
        
        Output: D_classe -> dict split by class, each set have the same class with the respective sigmoide value.
        """
        D_classe = {value: X_sig[X_sig[:,-1] == value]for value in np.unique(X_sig[:,-1])} #
        return D_classe
    
    """Faz novos subconjuntos com a intersecção dos grupos negativos com os grupos das classes
    def Intersec (D_neg, D_pos,D_classe): 
        D_inter_pos = {value: pd.merge(D_pos,D_classe[value]) for value in  (D_classe.keys()) }
        D_inter_neg = {value: pd.merge(D_neg,D_classe[value]) for value in  (D_classe.keys()) }
        return D_inter_pos, D_inter_neg
    """
    
    
    
    def _Intersec(self,D_neg, D_pos,D_classe): 
        """
        Intersection between the rows (events) in each set inside D_classe dict (each item has a respective set of data with the same class),
        and the set present in same demiplan. basically D_neg U D_class, or D- U D1, D+ U D1, D- U D1, D+ U D2. 
        
        Input: D_neg -> Set of data in negative demiplan, D_pos -> Set of data in positive demiplan, D_casse dict of set of
                events in the same class (each item in the dict is a set of data with the same class)
        
        Output:  D_inter_pos dict of set data with intersection between the demiplan side and the respective class
        so EX: D_inter_pos = D+ U D1, D+ U D2,..., D_inter_neg = D- U D1, D- U D2, .,..
        """
        D_inter_pos = {value: np.array([x for x in set(tuple(x) for x in D_pos) & set(tuple(x) for x in D_classe[value])]) for value in (D_classe.keys()) }
        D_inter_neg = {value: np.array([x for x in set(tuple(x) for x in D_neg) & set(tuple(x) for x in D_classe[value])]) for value in (D_classe.keys()) }
        return D_inter_pos, D_inter_neg
    
    
        """After fix and have the program work properly, i have to generalize the sigmoide to work with
        multiclass in a better way"""

    
        """
        Same thing as before but now we have the class with the sigmoide value instead of default value of equation without activation function
        """
       
    def _Intersec_v2(self, sig_neg, sig_pos): #D_cLASSE VEM DE GERAR GRUOPS 
        sig_inter_pos = {value: sig_pos[sig_pos[:,-1] == value]for value in np.unique(sig_pos[:,-1])} #
        sig_inter_neg = {value: sig_neg[sig_neg[:,-1] == value]for value in np.unique(sig_neg[:,-1])} 
        return sig_inter_pos, sig_inter_neg

    """Calcula a frequencia dividindo o cardinal de cada classe de cada conjunto pelo conjunto que ele pertence """
    def _frequency_heavi(self,D_inter_pos,D_inter_neg, D_neg, D_pos):
        
        """
        frequency with the heaviside function, divide the data in each intersection and the total cardinal of demiplan
        
        Input: D_inter_pos -> dict of all classes intersection and the positive Demiplan
               D_inter_neg -> dict of all classes intersection and the negative Demiplan
               D_neg -> set of negative values
               D_pos -> set of positive values
        
        Output: f_pos -> total frequency of events in positive demiplan
                f_neg -> total frequency of events in negative demiplan
        """
        Card_pos = len(D_pos)
        Card_neg = len(D_neg)
        if (Card_pos) == 0:   Card_pos =  10**-12
        if (Card_neg) == 0 :  Card_neg =  10**-12   
        f_pos = {value: len(D_inter_pos[value])/Card_pos for value in (D_inter_pos.keys())}
        f_neg = {value: len(D_inter_neg[value])/Card_neg for value in (D_inter_neg.keys())}
        return f_pos, f_neg
    
    def _frequency_sig(self,sig_inter_pos,sig_inter_neg, sig_neg, sig_pos):
        
        """
        frequency with the sigmoide function, divide the data in each intersection and the total cardinal of demiplan
        
        Input: sig_inter_pos -> dict of all classes intersection and the positive Demiplan
               sig_inter_neg -> dict of all classes intersection and the negative Demiplan
               D_neg -> set of negative values
               D_pos -> set of positive values
        
        Output: f_pos -> total frequency of events in positive demiplan
                f_neg -> total frequency of events in negative demiplan
        """
        f_pos = {value: np.sum(sig_inter_pos[value][:,0])/np.sum(sig_pos[:,0]) for value in sig_inter_pos.keys()}
        f_neg = {value: np.sum(sig_inter_neg[value][:,0])/np.sum(sig_neg[:,0]) for value in sig_inter_neg.keys()}
        return f_pos, f_neg

    """Calcula a entropia parcial de cada lado da reta, se a frequencia for 0 substitui por 10^-12"""
    #freq1*log(freq1)+freq2*log(freq2)...
    def __parcial_entropy(self,f_pos, f_neg):
     
        """
        Calculation of parcial entropy of S+ and S-
        
        Input: f_pos -> dict of positive frequencies of all classes,
               f_neg -> dict of negative frequencies of all classes
               
        Output: S_a_pos -> Scalar value of Positive Entropy
                S_a_neg -> Scalar value of Negative Entropy
        """
        #start = time.time()
        S_a_pos = 0 #f_pos e f_neg e um dicionario com valores
        S_a_neg = 0
        f_pos_list, f_neg_list = list(f_pos.values()), list(f_neg.values())#Em lista para facilitar os cálculos
        for i in  (f_pos_list):
            if (i == 0):  i = 10**-12
            S_a_pos = S_a_pos + (np.dot(i,np.log(i)))
        S_a_pos = - S_a_pos 
        for j in (f_neg_list):
            if (j == 0):   j = 10**-12
            S_a_neg = S_a_neg + (np.dot(j,np.log(j)))
        S_a_neg = -S_a_neg
        #print("time", end- start)
        return S_a_pos, S_a_neg
    
    
    """Calculated the total entropy Heavi side Function"""
    def _total_entropy_heavi(self,f_pos, f_neg, D_pos, D_neg, X):
        """
        Input: f_pos -> positive frequencies of each class in a dict
               f_neg -> negative frequencies of each class in a dict
               D_pos -> Set of all events presents in negative demiplan
               D_neg -> Set of all events presents in positive demiplan
               
        Output: S_a_pos -> Scalar positive frequencies
                S_a_neg -> Scalar negative frequencies
                S_a_Total -> Scalar of total entropy
        """
        S_a_pos, S_a_neg = self.__parcial_entropy(f_pos, f_neg)
        S_a_total = ((len(D_neg)* S_a_neg) + (len(D_pos)*S_a_pos))/len(X)
        return S_a_pos,S_a_neg, S_a_total
    
        """Calculated the total entropy Sigmoide Function"""
    def _total_entropy_sig(self,f_pos, f_neg, sig_pos, sig_neg):
        """
        Input: f_pos -> positive frequencies of each class in a dict
               f_neg -> negative frequencies of each class in a dict
               sig_pos -> Set of all events presents in negative demiplan with the respective sigmoide value
               Sig_neg -> Set of all events presents in positive demiplan with the respective sigmoide value
               
        Output: S_a_pos -> Scalar positive frequencies
                S_a_neg -> Scalar negative frequencies
                S_a_Total -> Scalar of total entropy
        """
        
        S_a_pos, S_a_neg = self.__parcial_entropy(f_pos, f_neg)
        if (sig_pos.size) == 0:   
            sig_pos_sum =  10**-12
        else: sig_pos_sum = np.sum(sig_pos[:,0])
        if (sig_neg.size) == 0:   
            sig_neg_sum =  10**-12
        else: sig_neg_sum = np.sum(sig_neg[:,0])
        N = sig_pos_sum+sig_neg_sum
        S_a_total = ((sig_neg_sum* S_a_neg) +(sig_pos_sum*S_a_pos))/N
        #sig_pos_sum* S_a_pos/N
        return  S_a_pos,S_a_neg, S_a_total
    
    "Recupera a classe com mais aparições em cada folha"
    def __dominant_class(self,D_pos, D_neg): #Alterado
        #label = str(y_train.columns.values[0]) # Fazer try catch error if the line its very bad
        #print("teste")
       #if len(D_neg) !=0:
        try:
            dominant_pos = np.bincount(D_pos.astype(int)[:, -1]).argmax()
        except ValueError:
            dominant_pos = 10000
        #else: dominant_pos = 5
        #if len(D_pos) !=0:
        try:
             dominant_neg = np.bincount(D_neg.astype(int)[:, -1]).argmax()
        except ValueError:
             dominant_neg = 10000
        #else: dominant_neg = 5
        return dominant_pos, dominant_neg
        
    "Gera uma imagem png com a estrutura da arvore (Old Version)"
    def Dot_Tree(self,root):
        """
        root -> first Node of the binary Tree, trough it its possible to access all the structure and subnodes.
        """
        DotExporter(root).to_dotfile('arvore_total.dot')
        Source.from_file('arvore_total.dot')
        render('dot', 'png', 'arvore_total.dot') 
    
    
    """As 4 funcções abaixo são todas para geração da nova arvore somente coma as informações relevantes"""
    def __nodenamefunc(self,model):
        if model.is_leaf == False :
           return '|| Id: %.3f || Entropia:  %s  ||N º Elementos:  %s  || Classe:  %s ||' % (model.name["id"],model.name["Entropia"], len(model.name["Data"]), model.name["classe"])
        elif model.is_leaf == True :
            return '|| Id: %.3f || N º Elementos:  %s || Entropia: %3s || Classe: %s ||' % (model.name["id"], len(model.name["Data"]), model.name["Entropia"],model.name["classe"])
       
    #Fiz uma abordagem utilizando o id mais posso acrescentar 000000000000.1 na entropia apra mudar 
    def __edgeattrfunc(self,model, child):
            return 'label="%s"' % (child.name["reta"])
        
    def __edgetypefunc(self,model, child):  
        return '--'    
    
    
    def Dot_Tree2(self,model):
        UniqueDotExporter(model, graph="graph",
                      nodenamefunc=self.__nodenamefunc,
                      nodeattrfunc=lambda model: "shape=rect",
                      edgeattrfunc= self.__edgeattrfunc,
                      edgetypefunc=self.__edgetypefunc).to_dotfile('arvore_totalnova.dot') 
        Source.from_file('arvore_totalnova.dot')
        render('dot', 'png', 'arvore_totalnova.dot') 
       
    """The first optmizer used to find the best line to divide the data, its a brute force method thats consisti in
       generate lines in a random way, and based on numbers of try, we will find the best line"""
    def __monte_carlo(self,X,y): #Aqui x_train possui a classe concatenada
    
        """
        Input: X -> numpy arrray of all atributes
           y -> numpy array with all classes
           
        Output: best_params_calc -> Dict with values of frequency and entropies of the last node dataset split
                list_S -> List of Minimum entropy found en each node split (for decrease plot of entropy over iteration)
                list_r -> Capture of the iteration that have found the minimum entropy (for decrease plot of entropy over iter)
                best_a -> Best vector found that give us the lowest entropy for a given dataset
                f_pos, f_neg -> Scalar value that Calculated frequency result of the current vecotr a found
                D_pos, D_neg -> Set's of each demiplan thats contain the values that belong
                S_a_pos, S_a_neg -> Scalar value, Entropy of each side of the demiplan for the current vector a of separation
    
        """
        list_S = []
        list_r =[]
        S_min = 10**20
        for r in range (1,self.nb_iter):
            a = self._random_a(X)
            _, _ , _ , _ , _ , S_a_pos, S_a_neg,S_a_total = self._Line(X,a)
            if (S_a_total < S_min):
                S_min = S_a_total #Agora tambem vou pegar as entropias parciais para utilizalas na arvore
                S_a_pos_min = S_a_pos
                S_a_neg_min = S_a_neg
                best_a = a
                r_min = r #iteracao que obteve o melhor resultado
                list_S.append(S_min)
                list_r.append(r_min)
        D_classe , f_pos, f_neg, D_pos, D_neg, S_a_pos, S_a_neg,S_a_total = self._Line(X,best_a)
        best_params_calc = {'f_neg' : f_neg, 'f_pos' : f_pos, 'S_a_pos' : S_a_pos, 'S_a_neg' : S_a_neg, 'S_total' : S_a_total }
        #np.sum(D_classe[1][:,0])
        return  best_params_calc, list_S,list_r,best_a,f_pos, f_neg, D_pos, D_neg, S_a_pos, S_a_neg  
    
    """Return Leaf with the highest entropy value if satisfy de function stop_leaf"""   
    def __readable_leaf(self,root,nb_branch): # IOld   def __stop_leaf(self,leaf, grid_entropy, grid_leaf):
         """
         Input : root -> Structure of Binary Tree, nb_branch -> Number of branches in the binary tree
         
         
         Output: S_min -> (Scalar) Go through each leaf and find the node with the lowest entropy and get the entropy value
                 new_leaf -> (Node)return the node with lowest entropy
                 nb_current -> (Scalar) Current value of number of branches
            """
         S_min = math.inf
         new_leaf = 0
         leaf_min = 0
         for leaf in root.leaves:
             if (self.__stop_leaf(leaf) == False):
                 if leaf_min < leaf.name["Entropia"]:
                    leaf_min = leaf.name["Entropia"]
                    new_leaf = leaf
                    S_min = leaf_min
         nb_branch = nb_branch + 1
         return S_min,new_leaf,nb_branch
     
    "Verifica se a folha esta em condições de ser separada"
    def __stop_leaf(self, leaf):
            """
            Input: leaf -> (Node) Current leaf
            
            Output: (bool) True or false if the node can be separated
            """
            if np.unique(leaf.name["Data"][:,-1]).size == 1:
                return True
            if(leaf.name["Data"].shape[0] <= leaf.name["Data"].shape[0]  * self.nb_leaf): 
                #print("aaaa", (leaf.name["Data"].shape[0]  * self.nb_leaf))#Passo o tamanho do banco de acordo com uma porcentagem dos dados
                return True
            else: 
               # print("ffffsdf", (leaf.name["Data"].shape[0]* self.nb_leaf))
                return False
                
    
         
    "Arvore binaria do dataset"
    def __New_Tree(self, X, y):
        
        """
        Input: X -> (np array) Input Train attributes, y -> (np array) Input train class
        
        
         Output: best_params_calc -> Dict with values of frequency and entropies of the last node dataset split
                list_S -> List of Minimum entropy found en each node split (for decrease plot of entropy over iteration)
                list_r -> Capture of the iteration that have found the minimum entropy (for decrease plot of entropy over iter)
                rooot -> (node) Binary Tree structure
                nb_branch -> (Scalar) Number of leafs of the binary Tree Structure
        """
        best_params_calc_list = []
        list_s_list = []
        list_r_list =[]
        nb_branch = 0
        nb_branch_max = self.nb_branch_max
        df = np.concatenate((X,y),axis=1) #Estou juntando os atributos com a classe de predicao
        rooot = NOD({"id": random(),"Data" :df,"Entropia": 100,"frequencia":0.1, "classe":90,"divisao": "centro"})
        S_min,new_leaf,nb_branch = self.__readable_leaf(rooot,nb_branch)
        if (self.optimizer == "Monte_Carlo"): #Depois resolver de maneira mais elegante
            while  ((S_min !=math.inf and nb_branch < nb_branch_max)):  #(S_min !=math.inf and root.height < 10)
                    best_params_calc,list_s,list_r,best_a,f_pos, f_neg,D_pos,D_neg,S_pos_min, S_neg_min = self.__monte_carlo(new_leaf.name["Data"],y) 
                    pos_classe, neg_classe = self.__dominant_class(D_pos, D_neg)
                    new_leaf.name["vetor"] = best_a
                    new_leaf.children = [NOD({"id": random(), "Data" : D_pos,"Entropia" :S_pos_min,"frequencia": f_pos, "classe": pos_classe, "divisao": "esquerda", "reta" : "> 0"}),NOD({"id": random(), "Data" : D_neg,"Entropia" :S_neg_min,"frequencia": f_neg, "classe": neg_classe, "divisao": "direita", "reta" : "<= 0" })]
                    treino =  new_leaf.name["Data"] #Vai ser utilizado para prever todos pontos do plano, e mostrar a reta
                    #plot_clf(new_leaf, treino['classe'], treino)
                    S_min,new_leaf,nb_branch = self.__readable_leaf(rooot,nb_branch)
                    best_params_calc_list.append(best_params_calc)
                    list_r_list.append(list_r)
                    list_s_list.append(list_s)
                    
        return best_params_calc_list,list_s_list, list_r_list,rooot, nb_branch

    def fit(self, X, y):
        _,_,self.root, self.total_branches = self.__New_Tree(X, y)
        
    
    "Calculo da funcao de psi dinamico para n atriburtos"    
    
    def _calcule_psi(self,row,a):
        pass

    def __find_label(self,i,threshold):
        node = self.root
        while node.is_leaf == False: 
              node = self.__next_node(node,i,threshold)
        return node.name["classe"]
    
    "Percorre a arvore com os dados de testes ate a folha"
    def __next_node(self,node,row,threshold): #WI will create a variable Condition that will have the treshhold to each actvatioin fucntion (Without activation is equal 0, sigmoide is equal 0.5)
        a = node.name["vetor"]
        psi = self._calcule_psi(row,a)
        for filho in node.children:
            if (psi <= threshold):
                if filho.name["divisao"] == "direita" :
                    node = filho 
                else: continue
            elif (psi >threshold):
                if  filho.name["divisao"]== "esquerda":
                     node = filho
                else: continue
        return node 
    
    "Classificar os Dados"
    def predict(self,x_teste, threshold):
        Predict = []
        for i in x_teste:
             label = self.__find_label(i,threshold)
             Predict.append(label)
        return Predict
        # ,Entropia,folhas,galhos,s,label, X_train, y_train,X_test, y_test
        #self, optimizer='Monte_Carlo', nb_iter = 1000, nb_entropy=10**-10 , nb_leaf = 15, nb_branch_max=10 
   
    
   
   
    def __Evaluate_Metrics_Binary(self,Predict, y_test):#Sao 2 listas
        #viz_confusion(Predict, real)
        TN, FP, FN, TP = confusion_matrix(y_test, Predict).ravel()                        
        print("------------Result Metrics----------------------------")
        print ("TN: ", TN )
        print ("FP: ", FP )
        print ("FN: ", FN )
        print ("TP: ", TP )
        PPV = TP/(TP+FP)
        print("Precision of Prediction: %f" %PPV)
        ACC = (TP+TN)/(TP+FP+FN+TN)
        print("Overall accuracy of Prediction: %f" %ACC)
        TPR = TP/(TP+FN)
        print("Recall of Prediction: %f" % TPR)
        print("------------------------------------------------------")
        return ACC, PPV, TPR
        
    
    "Metricas para avaliar o modelo criado tal que y > 2"
    def __Evaluate_metrics_multi_class(self,Predict, y_test):
        #label = str(y_train.columns.values[0])
        multi_cross = confusion_matrix(y_test, Predict)
        print(multi_cross) 
        FP = multi_cross.sum(axis=0) - np.diag(multi_cross)  
        FN = multi_cross.sum(axis=1) - np.diag(multi_cross)
        TP = np.diag(multi_cross)
        TN = multi_cross.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        FP = FP.sum()
        FN = FN.sum()
        TP = TP.sum()
        TN = TN.sum()
        PPV = TP/(TP+FP)
        ACC = TP/multi_cross.sum().sum()
        TPR =1
        print("Overall accuracy: ", ACC)
        return ACC , PPV, TPR
       
    def __Eval_bin_or_multi(self,y_size,Predict, y_test):
        #y_size = len(set(y_test)) #Convert to set to get uniqeu values only, then get len to get toal uniqeu values
        if y_size == 2:
           return self.__Evaluate_Metrics_Binary(Predict,y_test)
        elif y_size > 2:
            return self.__Evaluate_metrics_multi_class(Predict, y_test)
        
    def __viz_confusion(self,Predict, real):
        cm = confusion_matrix(real, Predict)
        cmd = ConfusionMatrixDisplay(cm)
        cmd.plot()
        cmd.ax_.set(xlabel='Predicted', ylabel='True')  
        #st.pyplot()
        return cmd
    
    #Se eu quiser colocar os dados com valoers default antes dos dados sek valores default é so colocar um asterisco antes.
    def grid_search(self,X, y, x_test, y_test, nb_entropy_list = [10**-10], nb_leaf_list = [15]  ,nb_branch_max_list = [10]):
        """ Gerar parametros para serem utilizados na geracao da arvore de forma a encontrar o melhor conjunto de parametros"""
        params = []
        i = 1
        for grid_entropy in nb_entropy_list:      
            self.nb_entropy = grid_entropy
            for grid_leaf in nb_leaf_list:
                self.nb_leaf = grid_leaf
                for grid_brench in nb_branch_max_list:
                    self.nb_branch_max = grid_brench
                    # self, x, y
                    best_params_calc,list_s, list_r,self.root, self.total_branches  = self.__New_Tree(X,y)
                   # print("fffff", (X.shape[0] * self.nb_leaf))
                    #nb_branch = nb_branch - 1
                    print ("Total Number of Combinations:{} ".format(len(nb_entropy_list)*len(nb_branch_max_list)*len(nb_leaf_list)))
                    print(" - ### Starting {} Iteration Of Training ###".format(i))
                    if (self.activation == 'sigmoide'):
                        Predict = self.predict(x_test, 0.5) #0.5 and 0 are the treshhold to decide what demiplan a event its
                    if (self.activation == 'heaviside'):
                        Predict = self.predict(x_test, 0)
                    accuracy = self.__Eval_bin_or_multi(len(np.unique(y)),Predict, y_test)
                    #params.append([self.root, self.total_branches, grid_entropy, grid_leaf, accuracy])
                    params.append([best_params_calc,list_s,list_r,self.root, self.total_branches, self.nb_entropy, self.nb_leaf,self.nb_branch_max, accuracy]) 
                    print(" - ### {} Iteration Of Training Finished ###".format(i))
                    print("------------------------------------------------------")
                    i = i+1
        best_params = sorted(params, key = lambda x: x[8], reverse=True)
        print("Melhores parametros Encontrados:")
        print("Entropia: ", best_params[0][5])
        print("% Of Data: ", best_params[0][6]) #Deopis arrumar para passar % em vez de quantidade bruta
        print("Nº Branchs: ", best_params[0][4])
        #plot_clf(best_params[0][0], real_in_sample, X_train)
        self.root = best_params[0][3]
        if (self.activation == 'sigmoide'):
            Predict = self.predict(x_test, 0.5) #0.5 and 0 are the treshhold to decide what demiplan a event its
        if (self.activation == 'heaviside'):
            Predict = self.predict(x_test, 0)
        print("Resultados Out-Sample Com os melhores parametros")
        acc, pre, rec = self.__Eval_bin_or_multi(len(np.unique(y)),Predict, y_test)
        cmd = self.__viz_confusion(Predict, y_test)
       #plot_clf(best_params[0][0], y_test, X_test)
        return best_params_calc,list_s,list_r,acc, pre, rec , cmd, best_params
     
class P1(EntropyClassifier):

 
      def _Line(self,X,a):
          super()._Line
          if( self.activation == 'heaviside'):
             #print("utilizado heaviside P1") 
             D_neg, D_pos = self._gerar_reta_p1(X,a)
             D_classe = self._groups_heavi(X)
             D_inter_pos, D_inter_neg = self._Intersec(D_neg, D_pos, D_classe)
             f_pos, f_neg =  self._frequency_heavi(D_inter_pos,D_inter_neg, D_neg, D_pos)
             S_a_pos,S_a_neg, S_a_total =  self._total_entropy_heavi(f_pos, f_neg, D_pos, D_neg, X)
          if( self.activation == 'sigmoide'):
             #print("utilizado sigmoide P1") 
             D_neg, D_pos,sig_neg, sig_pos,sig_func = self._gerar_reta_p1_sigmoide(X,a)
             D_classe = self._groups_sig(sig_func)
             sig_inter_pos, sig_inter_neg = self._Intersec_v2(sig_neg, sig_pos)
             f_pos, f_neg =  self._frequency_sig(sig_inter_pos, sig_inter_neg, sig_neg, sig_pos)
             S_a_pos,S_a_neg, S_a_total =  self._total_entropy_sig(f_pos, f_neg, sig_pos, sig_neg)
          return D_classe,f_pos, f_neg, D_pos, D_neg, S_a_pos, S_a_neg,S_a_total
         
   
      def _random_a(self,X):
            super()._random_a
            a = (np.random.uniform(-1/2,1/2,((X.shape[1],1)))).T#.uniform(-1/2,1/2)) np.random.uniform(start,stop,(rows,columns))
            #a = np.array([[-0.436667, 0.211111, -0.000000]])
            #a = np.array([[-0.152604, 0.442521,-0.465418]])
            #a = np.array([[-1.5350658989393161,0.5815000613075213, -0.8135463592809872]])
            new_a = self._norm_vector(a)
            return new_a
        
            """ 
       def gerar_reta_p1 (X_train,y_train,a):
            D_neg = []
            D_pos =  []
            for j in range (len(X_train)):
                psi = a[0]
                for  i in range (len(X_train.columns)-1): #X_train pois nao vou considerar a classe para multipolicacao
                    psi = psi + a[i+1]* X_train.iloc[j,i]
                if (psi <= 0): 
                        D_neg.append(X_train.iloc[j,:]) 
                else:
                        D_pos.append(X_train.iloc[j,:])
            print("RR")
            D_neg, D_pos = reshape_Data(D_neg, D_pos)
            print("d")
            return D_neg, D_pos #Retorna dois dataframes (Not a dict of dataframes, a Dataframe itself)
             """  
      def _gerar_reta_p1(self,X,a):
           id_neg = np.where(np.add(a[0,0],np.dot(a[0,1:], X[:,0:-1].T)) <= 0)
           id_pos = np.where(np.add(a[0,0],np.dot(a[0,1:], X[:,0:-1].T)) > 0)
           D_neg = X[id_neg]
           D_pos = X[id_pos]
           return D_neg, D_pos 
       
      def _gerar_reta_p1_sigmoide(self,X_train,a):
        reta = np.add(a[0,0],np.dot(a[0, 1:], X_train[:, 0:-1].T)) 
        func = lambda v: self._sigmoid(v)
        sig_func = np.array([func(s_j) for s_j in  reta])
        sig_func = np.column_stack((sig_func,X_train[:, -1])) #Concateno meus val de sigmoides com a coluna com a classe equivalente
        id_neg = np.where(sig_func[:,0] < 0.5)
        id_pos = np.where(sig_func[:,0] >= 0.5)
        D_neg = X_train[id_neg]
        D_pos = X_train[id_pos]
        sig_neg = sig_func.copy()
        sig_pos = sig_func.copy()
        np.subtract(1, sig_neg[:,0], out=sig_neg[:,0])
        return D_neg, D_pos,sig_neg, sig_pos,sig_func  
    
      """ 
      def _gerar_reta_p1_sigmoide(self,X_train,a):
        reta = np.add(a[0,0],np.dot(a[0, 1:], X_train[:, 0:-1].T)) 
        func = lambda v: self._sigmoid(v)
        sig_func = np.array([func(s_j) for s_j in  reta])
        sig_func = np.column_stack((sig_func,X_train[:, -1])) #Concateno meus val de sigmoides com a coluna com a classe equivalente
        sig_values = self._split(sig_func, sig_func[:,0] >= 0.5) #Separo pro maior e menor de 0.5
        id_neg = np.where(sig_func[:,0] < 0.5)
        id_pos = np.where(sig_func[:,0] >= 0.5)
        D_neg = X_train[id_neg]
        D_pos = X_train[id_pos]
        sig_neg = sig_values[1]
        sig_pos = sig_values[0]
        sig_neg_new = np.concatenate((sig_neg, sig_pos), axis=0)
        np.place(sig_neg_new[:,0], sig_neg_new[:,0] >= 0.5, 10**-20) #Este valoq que vou subistituir precisa ser menor que as entropias que vou setar no grid senao, ele slpita os dados mesmo quando possui somente uma classe
        sig_pos_new = np.concatenate((sig_pos, sig_neg), axis=0)
        np.place(sig_pos_new[:,0], sig_pos_new[:,0] < 0.5, 10**-20)
        #print("teset")
        return D_neg, D_pos,sig_neg_new, sig_pos_new,sig_func
       """
      def __find_label_heavi(self,i):
          node = self.root
          while node.is_leaf == False: 
                node = self.__next_node(node,i)
          return node.name["classe"]  
       
      def _calcule_psi(self,row,a):
          super()._calcule_psi
          psi = a[0]
          j=1
          psi = np.add.outer(a[0,0],np.dot(a[0,1:], row.T))
          if (self.activation == 'sigmoide'): #Ver outra maneira de fazer isto
               psi  = self._sigmoid(psi)
          return psi

    
class P2(EntropyClassifier):
         
        
      def _Line(self,X,y,a):
          super()._Line
          if( self.activation == 'heaviside'):
             print("utilizado heaviside P2")
             D_neg, D_pos = self._gerar_reta_p2(X,y,a)
             D_classe = self._groups_heavi(X, y)
             D_inter_pos, D_inter_neg = self._Intersec(D_neg, D_pos, D_classe)
             f_pos, f_neg =  self._frequency_heavi(D_inter_pos,D_inter_neg, D_neg, D_pos)
             S_a_pos,S_a_neg, S_a_total =  self._total_entropy_heavi(f_pos, f_neg, D_pos, D_neg, X)
          if( self.activation == 'sigmoide'):
             print("utilizado sigmoide P2")
             D_neg, D_pos,sig_neg, sig_pos,sig_func = self._gerar_reta_p2_sig(X,a)
             D_classe = self._groups_sig(sig_func)
             sig_inter_pos, sig_inter_neg = self._Intersec_v2(sig_neg, sig_pos)
             f_pos, f_neg =  self._frequency_sig(sig_inter_pos, sig_inter_neg, sig_neg, sig_pos)
             S_a_pos,S_a_neg, S_a_total =  self._total_entropy_sig(f_pos, f_neg, sig_pos, sig_neg)
          return D_classe,f_pos, f_neg, D_pos, D_neg, S_a_pos, S_a_neg,S_a_total
         
    
    
      def _index_in_list(self,a_list, index):
            if(index+1 < a_list.shape[1]):
                return True
            else: return False
      
            
      "Return vector a rand nums #a = (#X atributes)+1"
      def _random_a(self,X):
            super()._random_a #cardinal de a é definido por 3n
            a = []
            X_lenght = X.shape[1]-1 #Eu tiro aqui e coloco embaixo pq tenho que multiplicar somente os atributos, e depois eu volto pra contar com o a0 na soma final
            X_lenght += (X_lenght*2)+1
            for i in range (X_lenght):
                a = (np.random.uniform(-1/2,1/2,((i,1)))).T
            return a
        
      """
       def gerar_reta_p2(X_train,y_train,a):
            D_neg = []
            D_pos = []
            for j in range (len(X_train)):
                psi = a[0]
                f = 1
                for i in range (len(X_train.columns)-1):
                     psi = psi + a[f]*X_train.iloc[j,i]
                     f = f+1                                 #Se atributo = 2, ele sai dessa iteracao com f = 2
                for k in range (len(X_train.columns)-1):
                     psi = psi + a[f] * X_train.iloc[j,k]**2
                     f = f +1 
                     if (index_in_list(a, f) == True):
                         multi = 1 #Fazer uma condicional de que se a for out of range sair do loop
                         for t in range (len(X_train.columns)-1):
                             multi *= X_train.iloc[j,t]
                         multi = a[f] *multi 
                         f = f+1
                         psi = psi + multi
                if (psi <= 0): 
                        D_neg.append(X_train.iloc[j,:]) 
                else:
                        D_pos.append(X_train.iloc[j,:])
            D_neg, D_pos = reshape_Data(D_neg, D_pos)
            return D_neg, D_pos
        
        """
            
      def _gerar_reta_p2(self,X,y,a):
            z = X.shape[1] #Limite onde meu a vai multiplicar o vetor X de atributos
            j = 0
            psi = np.add(a[0,0],np.dot(a[0,1:z], X[:,0:-1].T)) #a0+a1*x1+a2*x2 ELE PEGA ATE 1 ANTES DO Z POR ISSO NAO PRECISO FAZER Z++
            #z = z+1
            while (self._index_in_list(a,z) == True):
                 psi = np.add(psi,np.dot(a[0,z], np.square(X[:,j].T)))
                 z = z +1
                 j = j +1
                 psi = np.add(psi, np.dot(a[0,z], np.prod(X[:,0:-1], axis=1)))
                 z = z +1
            psi = np.add(psi,np.dot(a[0,z], np.square(X[:,j].T)))
            id_neg = np.where(psi <= 0)
            id_pos = np.where(psi > 0)
            D_neg = X[id_neg]
            D_pos = X[id_pos]
            return D_neg, D_pos  
        
      def _gerar_reta_p2_sig(self,X_train,y_train,a, EPSI):
            z = X_train.shape[1] #Limite onde meu a vai multiplicar o vetor X de atributos
            j = 0
            psi = np.add(a[0,0],np.dot(a[0,1:z], X_train[:,0:-1].T)) #a0+a1*x1+a2*x2 ELE PEGA ATE 1 ANTES DO Z POR ISSO NAO PRECISO FAZER Z++
            #z = z+1
            while (self._index_in_list(a,z) == True):
                 psi = np.add(psi,np.dot(a[0,z], np.square(X_train[:,j].T)))
                 z = z +1
                 j = j +1
                 psi = np.add(psi, np.dot(a[0,z], np.prod(X_train[:,0:-1], axis=1)))
                 z = z +1
            psi = np.add(psi,np.dot(a[0,z], np.square(X_train[:,j].T)))
            func = lambda v: self._sigmoid(v,EPSI)
            sig_func = np.array([func(s_j) for s_j in  psi])
            sig_func = np.column_stack((sig_func,X_train[:, -1]))
            sig_values =  self._split(sig_func, sig_func[:,0] >= 0.5) #Separo pro maior e menor de 0.5
            id_neg = np.where(sig_func[:,0] < 0.5)
            id_pos = np.where(sig_func[:,0] >= 0.5)
            D_neg = X_train[id_neg]
            D_pos = X_train[id_pos]
            sig_neg = sig_values[1]
            sig_pos = sig_values[0]
            sig_neg_new = np.concatenate((sig_neg, sig_pos), axis=0)
            np.place(sig_neg_new[:,0], sig_neg_new[:,0] >= 0.5, 10**-20)
            sig_pos_new = np.concatenate((sig_pos, sig_neg), axis=0)
            np.place(sig_pos_new[:,0], sig_pos_new[:,0] < 0.5, 10**-20)
            return D_neg, D_pos,sig_neg_new, sig_pos_new,sig_func
                   
              
      def _calcule_psi(self,row,aç):
            super()._calcule_psi
            z = row.size+1 #Limite onde meu a vai multiplicar o vetor X de atributos
            j = 0
            psi = np.add(aç[0,0],np.dot(aç[0,1:z], row.T)) #a0+a1*x1+a2*x2 ELE PEGA ATE 1 ANTES DO Z POR ISSO NAO PRECISO FAZER Z++
            while (self._index_in_list(aç,z) == True):
                 psi = np.add(psi,np.dot(aç[0,z], np.square(row[j].T)))
                 z = z +1
                 j = j +1
                 psi = np.add(psi, np.dot(aç[0,z], np.prod(row)))
                 z = z +1
            psi = np.add(psi,np.dot(aç[0,z], np.square(row[j].T)))
            if (self.activation == 'sigmoide'):
               psi  = self._sigmoid(psi,1)
            return psi
                    
                    

    
    
    
    
    
    
    
    
    
              