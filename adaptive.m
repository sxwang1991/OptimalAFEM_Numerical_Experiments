function [eigv,eigf,eigv_a,eigf_a,Dof] = adaptive()

load matrix.mat
freeDof = freeDof'+1;
Dof = size(K,1);
nk=1:14;
nk2 = 21;
eigf = zeros(Dof,nk2);
[eigf(freeDof,:),eigv] = eigs(K(freeDof,freeDof), M(freeDof,freeDof),nk2,'sm');
Dof = size(K(freeDof,freeDof),1);
eigv= diag(eigv);
eigv = eigv(nk);
eigf = eigf(:,nk);
%norm_s = abs(conj(eigf.')*(M) *eigf);
%eigf= eigf./ (norm_s)^(1 / 2);
for itr = 1:14
norm_s = abs(conj(eigf(:,itr).')*(M) *eigf(:,itr));
eigf(:,itr)= eigf(:,itr) / (norm_s)^(1 / 2);
end
%% adjoint problem
eigf_a = zeros(Dof,nk2);
[eigf_a(freeDof,:),eigv_a] = eigs(K_adjoint(freeDof,freeDof), M(freeDof,freeDof),nk2,'sm');
eigv_a= diag(eigv_a);
eigv_a = eigv_a(nk);
eigf_a = eigf_a(:,nk);
for itr = 1:14
norm_a = abs(conj(eigf_a(:,itr).')*(M) *eigf_a(:,itr));
eigf_a(:,itr)= eigf_a(:,itr) / (norm_a)^(1 / 2);
end
end