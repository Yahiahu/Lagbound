"""Aggregated nonnegativity multipliers, shared by the solver and the certifier.
A level = (lab: vertex->cluster, pattern of (i,Q) pairs, sig >= 0 per pattern entry).
Constraint per entry:  h_iQ(X) = sum_{j in Q, j != i} d_j X_ij >= 0   (valid: X >= 0 on partitions).
Matrix form: S = sum sig_iQ * 0.5*(e_i w_Q^T + w_Q e_i^T) with w_Q = d*1_Q, minus its diagonal => zero diagonal.
<S,X> = sum sig_iQ h_iQ(X) >= 0.
"""
import numpy as np, scipy.sparse as sp
class Level:
    def __init__(self, lab, d, pi=None, pq=None):
        self.n=len(lab); self.lab=np.unique(lab,return_inverse=True)[1]; self.P=self.lab.max()+1; self.d=d
        self.W=sp.csr_matrix((d,(self.lab,np.arange(self.n))),shape=(self.P,self.n))   # P x n
        self.dense = pi is None      # dense: every (i,Q), sig stored as n x P array
        if self.dense:
            self.sig=np.zeros((self.n,self.P)); return
        self.pi=np.asarray(pi,np.int64); self.pq=np.asarray(pq,np.int64)
        self.self_=(self.lab[self.pi]==self.pq)
        self.sig=np.zeros(len(self.pi))
    def wsig(self):   # per-entry preconditioner weight source: d_i
        return self.d[:,None]*np.ones((1,self.P)) if self.dense else self.d[self.pi]
    def Smat(self):
        return sp.csr_matrix((self.sig,(self.pi,self.pq)),shape=(self.n,self.P))
    def apply(self,V):
        if self.dense:
            Y=self.W@V; Z=self.sig.T@V; sd=self.sig[np.arange(self.n),self.lab]*self.d
            return 0.5*(self.sig@Y+self.d[:,None]*Z[self.lab])-sd[:,None]*V
        Sg=self.Smat(); Y=self.W@V; Z=Sg.T@V
        sd=np.bincount(self.pi[self.self_],weights=self.sig[self.self_],minlength=self.n)*self.d
        return 0.5*(Sg@Y+self.d[:,None]*Z[self.lab])-sd[:,None]*V
    def h(self,V):      # assumes unit rows of V
        Y=self.W@V
        if self.dense:
            H=V@Y.T; H[np.arange(self.n),self.lab]-=self.d; return H
        return np.einsum('ij,ij->i',V[self.pi],Y[self.pq])-self.self_*self.d[self.pi]
    def dense_rows(self,c):   # rows c of S (exact-formula float evaluation), zero diagonal
        n=self.n
        if self.dense:
            S1=0.5*(self.W.T@self.sig[c].T).T; S2=0.5*self.d[c][:,None]*self.sig[:,self.lab[c]].T
            R=S1+S2; R[np.arange(len(c)),c]=0.0; return R
        Sg=self.Smat().tocsr()
        S1=0.5*(Sg[c]@self.W).toarray()                         # 0.5 sig[i,lab_j] d_j
        SgT=Sg.T.tocsr()                                         # P x n
        S2=0.5*self.d[c][:,None]*(SgT[self.lab[c]].toarray())   # 0.5 d_i sig[j,lab_i]
        R=S1+S2; R[np.arange(len(c)),c]=0.0
        return R
def neighbour_pattern(lab,u,v,n):
    lab=np.unique(lab,return_inverse=True)[1]; P=lab.max()+1
    pr=np.r_[np.arange(n),u,v]; pc=np.r_[lab,lab[v],lab[u]]
    pat=sp.csr_matrix((np.ones(len(pr)),(pr,pc)),shape=(n,P)); pat.sum_duplicates(); pat.sort_indices()
    return np.repeat(np.arange(n),np.diff(pat.indptr)), pat.indices.astype(np.int64)

class PairLevel:
    """Cluster-pair aggregated nonnegativity: for all clusters Q,R (sym):
       h_QR(X) = sum_{i in Q, j in R, i != j} d_i d_j X_ij >= 0.
    S = W^T Sig W with its diagonal removed (Sig symmetric P x P, >= 0);  <S,X> = sum_QR Sig_QR h_QR >= 0."""
    def __init__(self, lab, d):
        self.n=len(lab); self.lab=np.unique(lab,return_inverse=True)[1]; self.P=self.lab.max()+1; self.d=d
        self.W=sp.csr_matrix((d,(self.lab,np.arange(self.n))),shape=(self.P,self.n))
        self.dsq=np.bincount(self.lab,weights=d*d,minlength=self.P)
        self.sig=np.zeros((self.P,self.P)); self.dense=True; self.pair=True
    def apply(self,V):
        Y=self.W@V
        return self.d[:,None]*(self.sig@Y)[self.lab]-(np.diag(self.sig)[self.lab]*self.d**2)[:,None]*V
    def h(self,V):      # unit rows assumed
        Y=self.W@V; H=Y@Y.T; H[np.diag_indices(self.P)]-=self.dsq; return H
    def wsig(self): return np.ones((self.P,self.P))
    def dense_rows(self,c):
        R=self.d[c][:,None]*self.sig[self.lab[c]][:,self.lab]*self.d[None,:]
        R[np.arange(len(c)),c]=0.0; return R
