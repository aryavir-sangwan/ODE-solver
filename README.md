# ODE-solver
There are a number of ways to exactly solve a first order differential equation. To numerically solve a first order DE of the form u'=f(u,t), we require some numerical methods. Let's start with the Euler method. We are given an initial condition, u(t_0). We want to know what u(t) is, at some unspecified time t. Let t_0=a and t_n=b, where t_n=t. We can partition the interval [a,b] into n+1 equally spaced nodes, t_0, t_1,....,t_n. This is simple, given a step size h, h= (b-a)/n, then t_i = t_0 + i(h) for i=1,2,...n. Given a forward difference approximation of u' at t_i, $u'(t_i)= \frac{u(t_{i+1}) - u(t_i)}{h}$ for i=0,1,...n-1. This equation can be rearranged to be $$u(t_{i+1})= u(t_i) + (h)u'(t_i)$$

![Solver Page 1](images/Euler%20.png)
![Solver Page 2](images/Runge%20Kutta%20and%20IE2.png)
![Solver Page 3](images/Adams%20Bashforth%20and%20Adams%20Moulton.png)
# diffrentiation

![formulas page 1](images/FD.png)
# integration 

![integration page 1](images/Romberg.png)
