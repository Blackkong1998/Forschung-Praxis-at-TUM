
%Zero order hold collocation
%The input T is the sampling time 
T = 0.04;
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
%Collocation method
c1 = 1/2;

a_11 = 0.5;

b_1 = 1;

%important system matrix
A_1 = inv(eye(2)-T*a_11*A_con);
%discretization model
A_d = eye(2) + T*b_1*A_con*A_1;
B_d = T*b_1*A_con*A_1*T*a_11*B_con+T*b_1*B_con;
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
[Qf_d,~,~] = idare(A_d,B_d,Q,R,[],[]);      %ex
% Preprocessing
Uad_d = admissibleInputs(A_d,B_d,X_set,U_set,X_f,N); %ex

% Simulation for ex
X_log = zeros(n,steps);
U_log = zeros(2,steps);
X_pre = zeros(n,N,steps);
X_pre_true = zeros(n,N,steps);

x=x0;
%simulation for Gauss collocation
t_elapsed_ex = zeros(steps,1);
for i = 1:steps
    % Control Method: MPC
    tic
    U = MPC_opt(x,A_d,B_d,Q,Qf_d,R,Uad_d,N); % the control input sequence for predictions
    t_elapsed_ex(i) = toc;
    u = U(1:m);                                      % the optimal control input at current state
    
    % Prediction
    [A_,B_] = liftedModel(A_d,B_d,N);
    X =reshape( A_*x + B_*U, 2 ,N);
    
    %Open-loop performance:Apply the contol input sequence to the discretized model, 
    %and also transform the control input sequence to a continuous time sequence using ZOH and apply it to the true (continuous) system dynamics
   
%     Xpred=zeros(n,N+1);
%     Xpred(:,1)=x;
%     for j = 1:N
%         up = U(j,:);
%         [~,xp] = ode45(@(t,y) odes(t,y,[0;T],[up;up],A_con,B_con), [0;T], Xpred(:,j));
%         Xpred(:,j+1)=xp(end,:)';
%     end
%     Xpred=Xpred(:,2:end);

    % Log
    X_log(:,i) = x;
    U_log(:,i) = u;
    X_pre(:,:,i) = X;          %the prediction states of every MPC iteration in RK method
    %X_pre_true(:,:,i) = Xpred; %the open-loop prediction states in the true continuous system
    
    
    %Closed-loop performance:Apply an MPC iteration to retrieve an optimal (discrete time) control input u. 
    %This control input is then applied to the true (continuous) system in a ZOH fashion(constantly apply the
    %retrieved optimal control input for the duration of one sampling time T of the current discrete system). 
    % You will then retrieve the new current state.
    [~,x] = ode45(@(t,y) odes(t,y,[0;T],[u;u],A_con,B_con), [0;T], x);  % the next state by apply the control input at the current state to the real continuous system
    x=x(end,:)';               % the next state by apply the control input at the current state to the real continuous system 
end
figure(1)
%plot the true continuous system state of closed-loop performance
plot(X_log(1,:), X_log(2,:),'r-o','MarkerSize',5)
grid on

hold on
plot(x0(1),x0(2),'k*','MarkerSize',7)   %initial state
plot(0,0,'kx','MarkerSize',7)           %desired tracking point
ax = gca;
ax.FontSize = 16; 
set(gcf,'units','normalized','outerposition',[0 0 0.4 0.6])
xlim([-1.5,3.5]);
ylim([-1.5,2.5]);
xlabel('$x_1$','interpreter','latex','FontSize',24);
ylabel("$x_2$",'interpreter','latex','FontSize',24);
legend("Gauss collocation(ZOH)",'FontSize',20);
title('State trajectory of the true continuous system','FontSize',24);


% x_1 = linspace(0,3,steps+1);
% for i = 1:steps
%     figure(2)
%     line([x_1(i) x_1(i+1)],[U_log(1,i) U_log(1,i)],'LineWidth',2)
%     line([x_1(i+1) x_1(i+1)],[U_log(1,i) U_log(1,i+1)],'LineWidth',2)
%     hold on
%     ax = gca;
%     ax.FontSize = 36; 
%     xlabel("Time t",'FontSize',48);
%     ylabel("Control input u",'FontSize',48);
%     title("Zero-order hold control input",'FontSize',48)
% end

%Compute the invariant set

% X_f = ControlInvariantSet(A_d, B_d, Polyhedron('A',X_set.A,'b',X_set.b), Polyhedron('A',U_set.A,'b',U_set.b), 'maxItr',100);
% [X_f] = sortPolyhedron(X_f); % sort vertices of polyhedron because of tikz problem
% figure(3)
% plot(X_f, 'alpha', 0.1, X_f, 'alpha', 0.5)
% ax = gca;
% ax.FontSize = 36; 
% xlim([-2 4]);
% ylim([-2 3]);
% xlabel("$x_1$",'FontSize',24);
% ylabel("$x_2$",'FontSize',24);