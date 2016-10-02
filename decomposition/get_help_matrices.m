function [P, Q, M] = get_help_matrices(Abig, Bbig, d_)
% Computes help matrices P, Q and M

    M_hat = Bbig / Abig;
    sqrt_Abig = sqrtm(Abig);
    A_tilde = M_hat * sqrt_Abig;

    [P,S,Q] = svd(A_tilde); %S - from the biggest eigenvalues to the smallest
    
    % [P,S,Q] = svd(A_tilde, 'econ');
    %sum(sum(P'*P == 1)), sum(sum(Q'*Q == 1))
    
    U = P;
    V = sqrt_Abig \ Q;
    S_d_ = S(1:d_,1:d_);
    U_d_ = U(:,1:d_);
    V_d_ = V(:,1:d_);
    M = U_d_ * S_d_ * V_d_'; % here we have matrix M
     
    sqrt_S_d_ = sqrtm(S_d_);
    P = U_d_ * sqrt_S_d_;
    Q = V_d_ * sqrt_S_d_;
    
end
