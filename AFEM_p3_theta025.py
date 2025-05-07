from skfem import *
from skfem.visuals.matplotlib import *
from skfem.helpers import dot, grad
import numpy as np
import matlab.engine
engine = matlab.engine.start_matlab()

import scipy.io
from tools_bisect import bisect, uniformrefine

# parameters
intorder0 = 6 # for P3
itrMax = 91
theta = 0.25

epsilon = lambda x:  (10.)*((x[0]>=0.)&(x[1]>=0.)) + (10.)*((x[0]<0.)&(x[1]<0.)) \
                     + (1.)*((x[0]>=0.)&(x[1]<0.)) + (1.)*((x[0]<0.)&(x[1]>=0.))

@BilinearForm
def stiffness(u, v, w): 
    return epsilon(w.x)*(u.grad[0]*v.grad[0] + u.grad[1]*v.grad[1]) + 2*(u.grad[0]+u.grad[1])*v

@BilinearForm
def stiffness_adjoint(u, v, w): 
    return epsilon(w.x)*(u.grad[0]*v.grad[0] + u.grad[1]*v.grad[1]) - 2*(u.grad[0]+u.grad[1])*v



@BilinearForm
def mass(u, v, _):
	return u*v

p = np.array([[0., 1., 2., 0., 1., 2., 0., 1., 2.],
              [0., 0., 0., 1., 1., 1., 2., 2., 2.]], dtype=np.float64)
p = p - 1.
t = np.array([[1,4,0],
              [1,2,4],
              [3,0,4],
              [5,4,2],
              [3,4,6],
              [5,8,4],
              [7,6,4],    
              [7,4,8]], dtype=np.int64).T
belong = np.arange(t.shape[1],dtype=np.int64)
for _ in np.arange(2): 
    p,t,belong = uniformrefine(p,t,belong)
 
m = MeshTri(p,t)
   
m0 = MeshTri(p,t,sort_t=False)

e = ElementTriP3()



def eval_estimator(m, eigfuh,eigv):  
    
    eigfwh = eigfuh / eigv

    # interior residual
    basis = Basis(m, e, intorder=intorder0)

    grad_basis = basis.with_element(ElementVector(ElementDG(ElementTriP2()))) 
    w = {'grad_w': grad_basis.project(basis.interpolate(eigfwh).grad) }
    uh = basis.interpolate(eigfuh)

    @Functional 
    def interior_residual_vh(w): 
        h = w.h
        x, y = w.x
        (wxx, wxy), (wyx, wyy) = w['grad_w'].grad 
        return h ** 2 * abs(uh + epsilon(w.x)*(wxx + wyy) - 2*(w['grad_w'][0] + w['grad_w'][1])) ** 2 

    eta_K_vh = interior_residual_vh.elemental(grad_basis, **w) 
    
    # facet jump
    fbasis = [InteriorFacetBasis(m, e, side=i,intorder=intorder0) for i in [0, 1]]
    # w['v1'], w['v2']
    # jump of vh
    w = {'w' + str(i + 1): fbasis[i].interpolate(eigfwh) for i in [0, 1]}
    

    @Functional
    def edge_jump_vh(w):
        x = w.x
        xx = 0*x.copy()
        average = (x[:,:,0] + x[:,:,1] + x[:,:,2] + x[:,:,3])/4
        xx[:,:,0] = average
        xx[:,:,1] = average
        xx[:,:,2] = average
        xx[:,:,3] = average
        h = w.h
        n = w.n
        xIn = xx - (0.05*h)*n
        xOut = xx + (0.05*h)*n
        du1 = grad(w['w1'])
        du2 = grad(w['w2'])
        return h * abs((epsilon(xIn)*du1[0] - epsilon(xOut)*du2[0]) * n[0] + \
                       (epsilon(xIn)*du1[1] - epsilon(xOut)*du2[1]) * n[1]) ** 2

    eta_E_vh = edge_jump_vh.elemental(fbasis[0], **w)
    tmp_vh = np.zeros(m.facets.shape[1])
    np.add.at(tmp_vh, fbasis[0].find, eta_E_vh)
    eta_E_vh = np.sum(.5 * tmp_vh[m.t2f], axis=0)
    
    
    estimators = eta_K_vh + eta_E_vh
    
    return estimators

def eval_estimator_adjoint(m, eigfuh,eigv):  
    
    eigfwh = eigfuh / eigv

    # interior residual
    basis = Basis(m, e, intorder=intorder0)

    grad_basis = basis.with_element(ElementVector(ElementDG(ElementTriP2()))) 
    w = {'grad_w': grad_basis.project(basis.interpolate(eigfwh).grad) }
    uh = basis.interpolate(eigfuh)

    @Functional 
    def interior_residual_vh(w): 
        h = w.h
        x, y = w.x
        (wxx, wxy), (wyx, wyy) = w['grad_w'].grad 
        return h ** 2 * abs(uh + epsilon(w.x)*(wxx + wyy) + 2*(w['grad_w'][0]+w['grad_w'][1])) ** 2 

    eta_K_vh = interior_residual_vh.elemental(grad_basis, **w) 
    
    # facet jump
    fbasis = [InteriorFacetBasis(m, e, side=i,intorder=intorder0) for i in [0, 1]]
    # w['v1'], w['v2']
    # jump of vh
    w = {'w' + str(i + 1): fbasis[i].interpolate(eigfwh) for i in [0, 1]}
    
    @Functional
    def edge_jump_vh(w):
        x = w.x
        xx = 0*x.copy()
        average = (x[:,:,0] + x[:,:,1] + x[:,:,2] + x[:,:,3])/4
        xx[:,:,0] = average
        xx[:,:,1] = average
        xx[:,:,2] = average
        xx[:,:,3] = average
        h = w.h
        n = w.n
        xIn = xx - (0.05*h)*n
        xOut = xx + (0.05*h)*n
        du1 = grad(w['w1'])
        du2 = grad(w['w2'])
        return h * abs((epsilon(xIn)*du1[0] - epsilon(xOut)*du2[0]) * n[0] + \
                       (epsilon(xIn)*du1[1] - epsilon(xOut)*du2[1]) * n[1]) ** 2

    eta_E_vh = edge_jump_vh.elemental(fbasis[0], **w)
    tmp_vh = np.zeros(m.facets.shape[1])
    np.add.at(tmp_vh, fbasis[0].find, eta_E_vh)
    eta_E_vh = np.sum(.5 * tmp_vh[m.t2f], axis=0)
    
    
    estimators = eta_K_vh + eta_E_vh
    
    return estimators

def adaptive_theta_L2(eta, theta=0.5):
    isMarked = np.zeros(len(eta), dtype= bool)
    idx = np.argsort(eta)[-1::-1]
    x = np.cumsum(eta[idx])
    isMarked[idx[x <= theta*x[-1]]] = True
    isMarked[idx[0]] = True
    return isMarked

for itr in reversed(range(itrMax)):
    basis = Basis(m, e)

    K = asm(stiffness, basis)
    K_adjoint = asm(stiffness_adjoint, basis)
    M = asm(mass, basis)

    
    Eta = np.zeros((m.t.shape[1]))

    bdDof=basis.get_dofs().flatten()
    freeDof = basis.complement_dofs(bdDof).flatten()
    import scipy.io
    scipy.io.savemat("matrix.mat", mdict={'K': K,'K_adjoint': K_adjoint,'M':M,'bdDof': bdDof,'freeDof': freeDof})

    eigv,eigf,eigv_a,eigf_a,dof = engine.adaptive(nargout =5) 
    
    eigv = np.asarray(eigv)
    eigv_a = np.asarray(eigv_a)

    eigf = np.asarray(eigf)

    eigf_a = np.asarray(eigf_a)
    for kk in range(12): 
        eta = eval_estimator(m, eigf[:,kk], eigv[kk])
        eta_adjoint = eval_estimator_adjoint(m, eigf_a[:,kk], eigv_a[kk])
        Eta = Eta + (eta + eta_adjoint) 

    ev = eigv.flatten()
    print(itr,',',dof,',', Eta.sum(),',',ev[0].real,',',ev[1].real,',',ev[2].real,',',ev[3].real,',',ev[4].real,',',ev[5].real,',',ev[6].real,',',ev[7].real,',',ev[8].real,',',ev[9].real,',',ev[10].real,',',ev[11].real)
  


    if itr > 0:

        p,t,HB = bisect(m0.p,m0.t,adaptive_theta_L2(Eta, theta = theta))
        m0 = MeshTri(p,t,sort_t=False)
        m = MeshTri(p,t)

        from skfem.visuals.matplotlib import *
        draw(m)
        plt.savefig('Kellogg_P3_theta025_{}.pdf'.format(itrMax - itr),bbox_inches='tight')
        plt.close()




