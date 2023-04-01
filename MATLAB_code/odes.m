function rk = odes(t,y,T4,U,A_con,B_con)
%foramt long
U = interp1(T4,U,t);
rk(1) = A_con(1,1)*y(1) + A_con(1,2)*y(2) + B_con(1)*U;
rk(2) = A_con(2,1)*y(1) + A_con(2,2)*y(2) + B_con(2)*U;
rk = rk(:);