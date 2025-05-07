# -*- coding: utf-8 -*-

import numpy as np
from skfem import *    
from scipy.sparse import coo_matrix

def bisect(p,t,isMarkedElem,HB=None): 
    node = p.T 
    elem = t.T
    NN = node.shape[0]
    NT =  elem.shape[0] 
    localEdge = np.array([(1, 2), (2, 0), (0, 1)]) 
    totalEdge = elem[:, localEdge].reshape(-1, 2) 
    edge, i0, j = np.unique(np.sort(totalEdge, axis=-1),
            return_index=True,
            return_inverse=True,
            axis=0)
    NE = edge.shape[0]
    elem2edge = j.reshape(NT,3)
    edge2elem = np.zeros((NE, 4), dtype=np.int64)
    i1 = np.zeros(NE, dtype=np.int64)
    i1[j] = np.arange(3*NT, dtype=np.int64)
    edge2elem[:, 0] = i0 // 3
    edge2elem[:, 1] = i1 // 3
    edge2elem[:, 2] = i0 % 3
    edge2elem[:, 3] = i1 % 3
    neighbor = np.zeros((NT, 3), dtype=np.int64)
    neighbor[edge2elem[:, 0], edge2elem[:, 2]] = edge2elem[:, 1]
    neighbor[edge2elem[:, 1], edge2elem[:, 3]] = edge2elem[:, 0]
    isCutEdge = np.zeros((NE,), dtype=np.bool_)
    markedElem, = np.nonzero(isMarkedElem)
    while len(markedElem)>0:
        isCutEdge[elem2edge[markedElem, 0]]=True
        refineNeighbor = neighbor[markedElem, 0]
        markedElem = refineNeighbor[~isCutEdge[elem2edge[refineNeighbor,0]]]
    edge2newNode = np.zeros((NE,), dtype=np.int64)
    edge2newNode[isCutEdge] = np.arange(NN, NN+isCutEdge.sum()) #存储边上新节点的编号
    newNode = 0.5*(node[edge[isCutEdge,0],:] + node[edge[isCutEdge,1],:])
    node = np.concatenate((node, newNode), axis=0)
    elem2edge0 = elem2edge[:, 0]
    #------------------- Refine marked elements -------------------------------
    belong = np.zeros((4*NT,), dtype=np.int64) #一个单元最多3条边被二分
    
    if HB is None: 
        HB = np.arange(NT)
    
    for k in range(2):
        idx, = np.nonzero(edge2newNode[elem2edge0]>0)
        newNT = len(idx)
        if newNT == 0:
            break
        HB = np.concatenate((HB, HB[idx]), axis=0)
        L = idx
        R = np.arange(NT, NT+newNT)
        p0 = elem[idx,0]
        p1 = elem[idx,1]
        p2 = elem[idx,2]
        p3 = edge2newNode[elem2edge0[idx]]
        elem = np.concatenate((elem, np.zeros((newNT,3), dtype=np.int64)), axis=0)
        elem[L,0] = p3
        elem[L,1] = p0
        elem[L,2] = p1
        elem[R,0] = p3
        elem[R,1] = p2
        elem[R,2] = p0
        if k == 0:
            elem2edge0 = np.zeros((NT+newNT,), dtype=np.int64)
            elem2edge0[0:NT] = elem2edge[:,0]
            elem2edge0[L] = elem2edge[idx,2]
            elem2edge0[R] = elem2edge[idx,1]
        NT = NT + newNT
    return  node.T, elem.T, HB

def uniformrefine(p,t,belong): 
    node = p.T 
    elem = t.T
    N = node.shape[0]
    NT =  elem.shape[0] 
    localEdge = np.array([(1, 2), (2, 0), (0, 1)]) 
    totalEdge = elem[:, localEdge].reshape(-1, 2) 
    edge, i0, j = np.unique(np.sort(totalEdge, axis=-1),
            return_index=True,
            return_inverse=True,
            axis=0)
    NE = edge.shape[0]
    elem2edge = j.reshape(NT,3)
    node0 = node[edge[:,0],:] + node[edge[:,1],:]
    
    nodeNew = np.zeros((N+NE,2))
    elemNew = np.zeros((4*NT,3),dtype=np.int64) 
    belongNew = np.zeros((4*NT,),dtype=np.int64)
    
    nodeNew[0:N] = node
    nodeNew[N:N+NE,:] = (node[edge[:,0],:] + node[edge[:,1],:])/2;
    edge2newNode = np.arange(N,N+NE)
    t = np.arange(NT)
    p = np.zeros((NT,6))
    p[t,0:3] = elem[t,0:3];
    p[t,3:7] = edge2newNode[elem2edge[t,0:3]]
    elemNew[0:NT,:] = np.vstack((
                        p[t,0], 
                        p[t,5], 
                        p[t,4])).T
    elemNew[NT:2*NT,:] = np.vstack((
                        p[t,5], 
                        p[t,1], 
                        p[t,3])).T
    elemNew[2*NT:3*NT,:] = np.vstack((
                        p[t,4], 
                        p[t,3], 
                        p[t,2])).T
    elemNew[3*NT:4*NT,:] = np.vstack((
                        p[t,3], 
                        p[t,4], 
                        p[t,5])).T
    belongNew[0:4*NT] = np.hstack((belong,belong,belong,belong))
    return  nodeNew.T,elemNew.T,belongNew
