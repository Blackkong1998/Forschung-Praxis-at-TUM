
T = 0.03;
%The input T is the sampling time 
T_sim = 3;                %the simulation time
steps = floor(T_sim/T);   %the steps
N  = 10;                  %the prediction horizon
%continuous system dynamics
A_con = [0, 7.5; -1.5, -0.1];
B_con = [0.145;0.5];
%Dimension of state/input space
n = size(A_con,2);          % dimention of state space
m = size(B_con,2);          % dimention of input space
x0 = [2;2];             % initial state
%RK1 discretization model
%A = (I+delta_T*A_con),  B = delta_T*B_con
A_rk1 = eye(n) + T*A_con;    %System Matrix
B_rk1 = T*B_con;             % Input Matrix
% A_rk1 = A_d;
% B_rk1 = B_d;

%exact model
A_ex = expm(A_con*T);
B_ex = integral(@(t) expm(A_con.*t)*B_con, 0, T, 'ArrayValued', 1);
% Constraints A*x < b
U_set.A = -1;
U_set.b = 26;
%X_set.A = [-1,0;1,0;0,-1;0,1];
%X_set.b = [2.8;2.8;2.8;2.8];
X_set = Polyhedron('A',[1,0;0,1;-1,0;0,-1],'b',[2.8;2.1;1;1]);
% terminal constraint
X_f   = X_set;
% Optimization weights
Q  = diag([10,1]);
R  = 1;
[Qf_rk1,~,~] = idare(A_rk1,B_rk1,Q,R,[],[]);      %RK1
% Preprocessing
Uad_rk1 = admissibleInputs(A_rk1,B_rk1,X_set,U_set,X_f,N); %RK1
% Simulation for RK1
X_log = zeros(n,steps);
U_log = zeros(m,steps);
X_pre = zeros(n,N,steps);
X_pre_true = zeros(n,N,steps);

x=x0;
%simulation for RK1
t_elapsed_rk1 = zeros(steps,1);
for i = 1:steps
    % Control Method: MPC
    tic
    U = MPC_opt(x,A_rk1,B_rk1,Q,Qf_rk1,R,Uad_rk1,N); % the control input sequence for predictions
    t_elapsed_rk1(i) = toc;
    u = U(1:m);                                      % the optimal control input at current state
    
    % Prediction
    [A_,B_] = liftedModel(A_rk1,B_rk1,N);
    X =reshape( A_*x + B_*U, 2 ,N);
    
    %Open-loop performance:Apply the contol input sequence to the discretized model, 
    %and also transform the control input sequence to a continuous time sequence using ZOH and apply it to the true (continuous) system dynamics
   
    Xpred=zeros(n,N+1);
    Xpred(:,1)=x;
    for j = 1:N
        up = U(j,:);
        [~,xp] = ode45(@(t,y) odes(t,y,[0;T],[up;up],A_con,B_con), [0;T], Xpred(:,j));
        Xpred(:,j+1)=xp(end,:)';
    end
    Xpred=Xpred(:,2:end);

    % Log
    X_log(:,i) = x;
    U_log(:,i) = u;
    X_pre(:,:,i) = X;          %the prediction states of every MPC iteration in RK method
    X_pre_true(:,:,i) = Xpred; %the open-loop prediction states in the true continuous system
    
    
    %Closed-loop performance:Apply an MPC iteration to retrieve an optimal (discrete time) control input u. 
    %This control input is then applied to the true (continuous) system in a ZOH fashion(constantly apply the
    %retrieved optimal control input for the duration of one sampling time T of the current discrete system). 
    % You will then retrieve the new current state.
    [~,x] = ode45(@(t,y) odes(t,y,[0;T],[u;u],A_con,B_con), [0;T], x);   % the next state by apply the control input at the current state to the real continuous system
    x=x(end,:)';               % the next state by apply the control input at the current state to the real continuous system 
end
error = sum(abs(X_pre- X_pre_true),"all")/steps;
X_pre_true(X_pre_true > 2.80001) = NaN;
NaNs = nnz(isnan(X_pre_true));
fprintf('the number of constraint violation is:%f\n', NaNs);

% figure(1)
% %plot the time consumption for every MPC iteration
% deltat = linspace(0,3,75);
% t1 = reshape(t_elapsed_rk1,[75,5]);
% t_elapsed_rk1=sum(t1,2);
% plot(deltat,t_elapsed_rk1,'-','LineWidth', 4);
% hold on
% ax = gca;
% ax.FontSize = 16; 
% xlabel("Simulation time",'FontSize',24);
% ylabel("time consumption for each MPC iteration",'FontSize',24)
% title("Time Consumption of MPC")
% 
% figure(2)
% %plot the true continuous system state of closed-loop performance
% plot(X_log(1,:), X_log(2,:),'k-o','MarkerSize',5)
% % 
% %threshold = 2.8;
% moreThanThreshold = X_log(1,:) > 2.80001; % Logical indexes.
% % Extract those over the threshold into new arrays.
% over_x = X_log(1,moreThanThreshold);
% over_y = X_log(2,moreThanThreshold);
% % Now plot those over 2.8 with red stars over the first set of points.
% hold on; % Important so you don't blow the first plot away.
% % plot(over_x, over_y, 'r*', 'LineWidth', 2, 'MarkerSize', 9);
% % hold on
% % plot(x0(1),x0(2),'k*','MarkerSize',9)   %initial state
% % plot(0,0,'kx','MarkerSize',9)           %desired tracking point
% grid on
% ax = gca;
% ax.FontSize = 16; 
% xlim([-1.5,3.5]);
% ylim([-1.5,2.5]);
% xlabel('$x_1$','interpreter','latex','FontSize',24);
% ylabel("$x_2$",'interpreter','latex','FontSize',24);
% %legend("RK1");
% title("True continuous system closed-loop",'FontSize',24)

%Error computation

% fprintf('the error between RK1 MPC prediction states and true continuous time system states is:%3f\n', error);

%Compute the control invariant set
% X_f_rk = ControlInvariantSet( A_rk1, B_rk1, Polyhedron('A',X_set.A,'b',X_set.b), Polyhedron('A',U_set.A,'b',U_set.b), 'maxItr',100);
% X_f_ex = ControlInvariantSet( A_ex, B_ex, Polyhedron('A',X_set.A,'b',X_set.b), Polyhedron('A',U_set.A,'b',U_set.b), 'maxItr',100);
% [X_f_rk] = sortPolyhedron(X_f_rk); % sort vertices of polyhedron because of tikz problem
% [X_f_ex] = sortPolyhedron(X_f_ex);
% 
% figure(3)
% plot(X_f_ex, 'alpha', 0.1, X_f_rk, 'alpha', 0.5)
% hold on
% plotStateSpace (i, X_log,X_pre)
% ax = gca;
% ax.FontSize = 32; 
% legend('invariant set of exact model','invariant set of RK1 model','FontSize',20);
% xlim([-1.5,3.5]);
% ylim([-1.5,2.5]);
% xlabel("$x_1$",'FontSize',36);
% ylabel("$x_2$",'FontSize',36);
% title('the control invariant set','FontSize',24)


%Compute the invariant set
% sys_rk = LTISystem('A', A_rk1);
% X_f0_rk = sys_rk.invariantSet('X', Polyhedron('A',X_set.A,'b',X_set.b),'maxIterations',150);
% sys_ex = LTISystem('A', A_ex);
% X_f0_ex = sys_ex.invariantSet('X', Polyhedron('A',X_set.A,'b',X_set.b),'maxIterations',150);
% [X_f0_rk] = sortPolyhedron(X_f0_rk); % sort vertices of polyhedron because of tikz problem
% [X_f0_ex] = sortPolyhedron(X_f0_ex);
% 
% figure(4)
% plot(X_f0_ex, 'alpha', 0.1, X_f0_rk, 'alpha', 0.5)
% hold on
% plotStateSpace (i, X_log,X_pre)
% ax = gca;
% ax.FontSize = 32; 
% legend('invariant set of exact model','invariant set of RK1 model','FontSize',20);
% xlim([-1.5,3.5]);
% ylim([-1.5,2.5]);
% xlabel("$x_1$",'FontSize',36);
% ylabel("$x_2$",'FontSize',36);
% title('the invariant set','FontSize',24)