<h1 align="center">
<br> The k-Medoids Problem through Lagrangian Duality lens
</h1>
Repository for the course on Combinatorial Optimization at  <a href="https://www.cos.ufrj.br/" > PESC - Systems Engineering and Computer Science Program</a> from <a href="https://ufrj.br/" >UFRJ - Federal University of Rio de Janeiro</a>, taught by <a href="https://www.cos.ufrj.br/index.php/pt-BR/pessoas/details/18/2201-abiliolucena">Prof.  Abilio Pereira de Lucena Filho</a>.

Developed by Ronald Albert.
<h2 align="center">
The project
</h2>
This work was developed for the Combinatorial Optimization course, taught by Professor Abílio Lucena in 2023/P2 for the Computer Systems Engineering Program (PESC). In this work, Lagrangian relaxation will be applied to the analysis of clustering in the specific context of the k-medoids problem. More specifically, by usage of the subgradient method upper and lower bounds are gradually generated for bounding the objective function, with the hope that suchbounds eventually converge to the optimal value of the objective function.

It's entirely implemented in Julia, and all the results are available in the folder `results`.

<h2 align="center">
File list
</h2>
<ul>
    <li><h3>src/OptimalCluster.jl</h3></li>
    <p>Main module of the project where all other modules are imported.</p>
    <li><h3>src/HomogeneousCluster.jl</h3></li>
    <p>Module for implementing the $k$-medoids clustering problem, an approximate implementation is implemented with the subgradient method, as well as, an exact solution with the help of <strong>Gurobi</strong> solvers.</p>
</ul>
