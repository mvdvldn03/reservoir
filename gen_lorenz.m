function [output] = gen_lorenz(init_pos,sigma,beta,rho,tf,dt)
%GEN_LORENZ
%Generates Lorenz Trajectory based on initial position (init_pos),
%sig/rho/beta values, every dt seconds to a final time (t_f)

f = @(t,a) [-sigma*a(1) + sigma*a(2);
            rho*a(1) - a(2) - a(1)*a(3);
            -beta*a(3) + a(1)*a(2)];
[~,a] = ode45(f,[0:dt:tf], init_pos);

output = a;
end

