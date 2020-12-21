from subprocess import Popen, PIPE
from math import *
import random
import numpy as np
import random as pr
import csv
import itertools
import time
import math
import argparse
import subprocess
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
import sys
from rdkit.Chem import AllChem
from make_smile import zinc_data_with_bracket_original, zinc_processed_with_bracket
import sascorer
import pickle
import gzip
import networkx as nx
from rdkit.Chem import rdmolops
from load_model import loaded_model
from keras.preprocessing import sequence
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
import sys
from make_smile import zinc_data_with_bracket_original, zinc_processed_with_bracket
#from add_node_type import chem_kn_simulation, make_input_smile,predict_smile,check_node_type,node_to_add,expanded_node
def chem_kn_simulation(model,state,val):
    all_posible=[]
    end="\n"
    position=[]
    position.extend(state)
    total_generated=[]
    new_compound=[]
    get_int_old=[]
    for j in range(len(position)):
        get_int_old.append(val.index(position[j]))
    get_int=get_int_old
    x=np.reshape(get_int,(1,len(get_int)))
    x_pad= sequence.pad_sequences(x, maxlen=82, dtype='int32',
        padding='post', truncating='pre', value=0.)
    while not get_int[-1] == val.index(end):
        predictions=model.predict(x_pad)
        #print "shape of RNN",predictions.shape
        preds=np.asarray(predictions[0][len(get_int)-1]).astype('float64')
        preds = np.log(preds) / 1.0
        preds = np.exp(preds) / np.sum(np.exp(preds))
        next_probas = np.random.multinomial(1, preds, 1)
        next_int=np.argmax(next_probas)
        #a=predictions[0][len(get_int)-1]
        #next_int_test=sorted(range(len(a)), key=lambda i: a[i])[-10:]
        get_int.append(next_int)
        x=np.reshape(get_int,(1,len(get_int)))
        x_pad = sequence.pad_sequences(x, maxlen=82, dtype='int32',
            padding='post', truncating='pre', value=0.)
        if len(get_int)>82:
            break
    total_generated.append(get_int)
    all_posible.extend(total_generated)
    return all_posible

def predict_smile(all_posible,val):
    new_compound=[]
    for i in range(len(all_posible)):
        total_generated=all_posible[i]
        generate_smile=[]
        for j in range(len(total_generated)-1):
            generate_smile.append(val[total_generated[j]])
        generate_smile.remove("&")
        new_compound.append(generate_smile)
    return new_compound

def make_input_smile(generate_smile):
    new_compound=[]
    for i in range(len(generate_smile)):
        middle=[]
        for j in range(len(generate_smile[i])):
            middle.append(generate_smile[i][j])
        com=''.join(middle)
        new_compound.append(com)
    return new_compound

def expansion(node,model):
    st_time=time.time()
    state=node.state
    val=['\n', '&', 'C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F',
            '[C@@H]', 'n', '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]',
            's', 'Br', '/', '[nH]', '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]',
            '[nH+]', '\\', '[S@]', '5', '[N-]', '[n+]', '[S@@]', '[S-]', '6',
            '7', 'I', '[n-]', 'P', '[OH+]', '[NH-]', '[P@@H]', '[P@@]', '[PH2]',
            '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]', '[SH+]', '[O+]',
            '[s+]', '[PH+]', '[PH]', '8', '[S@@+]']
    all_nodes=[]
    end="\n"
    position=[]
    position.extend(state)
    total_generated=[]
    new_compound=[]
    get_int_old=[]
    for j in range(len(position)):
        get_int_old.append(val.index(position[j]))
    get_int=get_int_old
    x=np.reshape(get_int,(1,len(get_int)))
    x_pad= sequence.pad_sequences(x, maxlen=82, dtype='int32',
        padding='post', truncating='pre', value=0.)
    predictions=model.predict(x_pad)
    preds=np.asarray(predictions[0][len(get_int)-1]).astype('float64')
    preds = np.log(preds) / 1.0
    preds = np.exp(preds) / np.sum(np.exp(preds))
    sort_index = np.argsort(-preds)
    i=0
    sum_preds=preds[sort_index[i]]
    all_nodes.append(sort_index[i])
    while sum_preds<=0.95:
        i+=1
        all_nodes.append(sort_index[i])
        sum_preds+=preds[sort_index[i]]
    fi_time=time.time()-st_time
    node.check_childnode.extend(all_nodes)
    node.expanded_nodes.extend(all_nodes)

    return node,i


def addnode(node,m):
    val=['\n', '&', 'C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[C@@H]', 'n',
            '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]', 's', 'Br', '/', '[nH]',
            '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]', '[nH+]', '\\', '[S@]', '5', '[N-]', '[n+]',
            '[S@@]', '[S-]', '6', '7', 'I', '[n-]', 'P', '[OH+]', '[NH-]', '[P@@H]', '[P@@]',
            '[PH2]', '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]', '[SH+]', '[O+]', '[s+]',
            '[PH+]', '[PH]', '8', '[S@@+]']

    node.expanded_nodes.remove(m)
    added_nodes=[]
    added_nodes.extend(node.state)
    added_nodes.append(val[m])
    n=Node(state=added_nodes,parentNode=node)
    node.childNodes.append(n)
    return node,n

def simulation(chem_model,state):
    val=['\n', '&', 'C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[C@@H]', 'n',
            '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]', 's', 'Br', '/', '[nH]',
            '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]', '[nH+]', '\\', '[S@]', '5', '[N-]', '[n+]',
            '[S@@]', '[S-]', '6', '7', 'I', '[n-]', 'P', '[OH+]', '[NH-]', '[P@@H]', '[P@@]',
            '[PH2]', '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]', '[SH+]', '[O+]', '[s+]',
            '[PH+]', '[PH]', '8', '[S@@+]']
    all_posible=chem_kn_simulation(chem_model,state,val)
    generate_smile=predict_smile(all_posible,val)
    new_compound=make_input_smile(generate_smile)
    kao=[]
    try:
        m = Chem.MolFromSmiles(str(new_compound[0]))
    except:
        m=None
    if m!=None:
        try:
            logp=Descriptors.MolLogP(m)
        except:
            logp=-1000
        SA_score = -sascorer.calculateScore(MolFromSmiles(new_compound[0]))
        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(new_compound[0]))))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([ len(j) for j in cycle_list ])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6
        cycle_score = -cycle_length
        SA_score_norm=SA_score#(SA_score-SA_mean)/SA_std
        logp_norm=logp#(logp-logP_mean)/logP_std
        cycle_score_norm=cycle_score#(cycle_score-cycle_mean)/cycle_std
        score_one = SA_score_norm + logp_norm + cycle_score_norm
        score=score_one/(1+abs(score_one))
    else:
        score=-1000/(1+1000)
    return score,new_compound[0]


def Selectnode(node):
    ucb=[]
    for i in range(len(node.childNodes)):
        ucb.append(node.childNodes[i].wins/node.childNodes[i].visits+0.1*sqrt(2*log(node.visits)/node.childNodes[i].visits))
        m = np.amax(ucb)
        indices = np.nonzero(ucb == m)[0]
        ind=pr.choice(indices)
        s=node.childNodes[ind]
    return s


def Update(node,result):
    node.visits += 1
    node.wins += result
    return node


def write_to_csv(wfile,name):
    with open(str(name)+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter ='\n')
        writer.writerow(wfile)


class Node:

    def __init__(self, state,  parentNode = None):
        self.state = state
        self.parentNode = parentNode
        self.childNodes = []
        self.child=None
        self.wins = 0
        self.visits = 0
        self.depth=0
        self.expanded_nodes=[]
        self.check_childnode=[]


def MCTS(root):

    """initialization of the chemical trees and grammar trees"""
    run_time=time.time()
    rootnode = Node(state = root)
    start_time=time.time()
    """----------------------------------------------------------------------"""

    """global variables used for save valid compounds and simulated compounds"""
    valid_compound=[]
    all_simulated_compound=[]
    desired_compound=[]
    max_score=-100.0
    desired_activity=[]
    time_distribution=[]
    num_searched=[]
    current_score=[]
    depth=[]
    branch=0.0
    num_node=0.0
    allscore=[]
    allmol=[]
    tb=[]
    tn=[]
    """----------------------------------------------------------------------"""
    while time.time()-run_time<=153600:#600*256:
        node = rootnode
        """selection step"""
        #print ("node.childNodes:",len(node.childNodes),len(node.check_childnode),len(node.expanded_nodes))
        while node.childNodes!=[] and node.check_childnode!=[] and node.expanded_nodes==[]:
            node = Selectnode(node)
        #print ("selection:",node.state)
            #state.SelectPosition(node.position)
        #depth.append(len(node.state))
        if len(node.state)>=81:
            re=-1.0
            while node != None:
                Update(node,re)
                node = node.parentNode
        else:
            """------------------------------------------------------------------"""
            """expansion step"""
            """calculate how many nodes will be added under current leaf"""
            if node.check_childnode==[]:
                node,i=expansion(node,model)
                if i!=0:
                    branch+=1
                    num_node+=i
                m=random.choice(node.expanded_nodes)
                node,n=addnode(node,m)
                node=n
                score,mol=simulation(model,n.state)
                allscore.append(score)
                allmol.append(mol)
                depth.append(len(node.state))
            else:
                m=random.choice(node.expanded_nodes)
                node,n=addnode(node,m)
                node=n
                score,mol=simulation(model,n.state)
                allscore.append(score)
                allmol.append(mol)
                depth.append(len(node.state))
 
            """simulation"""
            #re=score/(1.0+abs(score))
            #node=Update(node,re)
            re=score
            """backpropation step"""
            while node!= None:
                node=Update(node,re)
                node=node.parentNode

    #print ("score=", allscore)
    #print ("mol=",allmol)
    tb.append(branch)
    tn.append(num_node)

    write_to_csv(allscore,'score')
    write_to_csv(allmol,'mol')
    write_to_csv(depth,'depth')
    write_to_csv(tb,'tb')
    write_to_csv(tn,'tn')





    return valid_compound


if __name__ == "__main__":
    #smile_old=zinc_data_with_bracket_original()
    #val,smile=zinc_processed_with_bracket(smile_old)
    #print (val)
    model=loaded_model()
    #acitivity_model=loaded_activity_model()
    valid_compound=MCTS(root=['&'])
