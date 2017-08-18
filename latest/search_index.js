var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#JacobiDavidson.jl-1",
    "page": "Home",
    "title": "JacobiDavidson.jl",
    "category": "section",
    "text": "This package implements the Jacobi-Davidson method for (generalized) non-Hermitian eigenvalue problems involving large, sparse matrices. It provides two methods jdqr and jdqz which iteratively generate a partial, approximate Schur decomposition for a matrix A or a matrix pencil (A B). The Schur vectors form an orthonormal basis for eigenspaces, and they can easily be transformed to eigenvectors as well."
},

{
    "location": "index.html#Jacobi-Davidson-versus-Arnoldi-1",
    "page": "Home",
    "title": "Jacobi-Davidson versus Arnoldi",
    "category": "section",
    "text": "Jacobi-Davidson can be particularly useful compared to Arnoldi when eigenvalues around a specific target tau in the complex plane are requested. For the standard eigenvalue problem, the Arnoldi method would expand the search space using v_n+1 = (A - tau I)^-1v_n. This linear system must be solved rather accurately in order to retain the Arnoldi decomposition.The Jacobi-Davidson method on the other hand is not a Krylov subspace method and does not rely on an accurate Arnoldi decomposition; rather it can be seen as a subspace accelerated, approximate Newton method. Each iteration it must solve a linear system as well, but the upside is that it does this only approximately. The basic premise being that there is no need to solve intermediate, approximate solutions in the Newton iterations to full precision.This means that Jacobi-Davidson can use a few steps of an iterative method internally, optionally with a preconditioner."
},

{
    "location": "index.html#Example-1",
    "page": "Home",
    "title": "Example",
    "category": "section",
    "text": "Let A and B be diagonal matrices of size n with A_kk = sqrtk and B_kk = 1  sqrtk. The eigenvalues of the problem Ax = lambda Bx are 1 dots n. The exact inverse of (A - tau B) is used as a preconditioner, namely a diagonal matrix P with P_kk = sqrtk  (k - tau). We implement these linear operators matrix-free:import Base.LinAlg.A_ldiv_B!\n\nfunction myA!(y, x)\n  for i = 1 : length(x)\n    @inbounds y[i] = sqrt(i) * x[i]\n  end\nend\n\nfunction myB!(y, x)\n  for i = 1 : length(x)\n    @inbounds y[i] = x[i] / sqrt(i)\n  end\nend\n\nstruct SuperPreconditioner{numT <: Number}\n  target::numT\nend\n\nfunction A_ldiv_B!(p::SuperPreconditioner, x)\n  for i = 1 : length(x)\n    @inbounds x[i] = x[i] * sqrt(i) / (i - p.target)\n  end\nendNext we call jdqz to solve the generalized eigenvalue problem for just 5 eigenvalues near 5000.1:using JacobiDavidson\nusing LinearMaps\n\nn = 10_000\ntarget = Near(5_000.1 + 0.0im)\nA = LinearMap{Float64}(myA!, n; ismutating = true)\nB = LinearMap{Float64}(myB!, n; ismutating = true)\nP = SuperPreconditioner(target.τ)\n\nschur, residuals = jdqz(A, B, \n    bicgstabl_solver(A, max_mv_products = 10, l = 2),\n    preconditioner = P,\n    testspace = Harmonic,\n    target = target,\n    pairs = 5,\n    ɛ = 1e-9,\n    min_dimension = 10,\n    max_dimension = 20,\n    max_iter = 100,\n    verbose = true\n)It finds the eigenvalues from 4998 to 5002. We can then plot the convergence history:using Plots\nplot(residuals, marker = :+, yscale = :log10, label = \"Residual norm\")which shows(Image: Convergence history)"
},

{
    "location": "solvers.html#",
    "page": "Correction equation",
    "title": "Correction equation",
    "category": "page",
    "text": ""
},

{
    "location": "solvers.html#Solvers-for-the-correction-equation-1",
    "page": "Correction equation",
    "title": "Solvers for the correction equation",
    "category": "section",
    "text": "At this point preconditioned GMRES and BiCGStabl(l) are available as iterative methods to solve the correction equation. Allocations for these methods are done only once during initialization."
},

{
    "location": "solvers.html#BiCGStab(l)-1",
    "page": "Correction equation",
    "title": "BiCGStab(l)",
    "category": "section",
    "text": "BiCGStab(l) is a non-optimal Krylov subspace method, but is of interest because it has a fixed amount of operations per iteration:solver = bicgstabl_solver(n, max_mv_products = 10, l = 2)"
},

{
    "location": "solvers.html#GMRES-1",
    "page": "Correction equation",
    "title": "GMRES",
    "category": "section",
    "text": "GMRES selects the minimal residual solution from a Krylov subspace. We use GMRES without restarts, since we assume only a few iterations are performed.solver = gmres_solver(n, iterations = 5)"
},

{
    "location": "solvers.html#Preconditioning-1",
    "page": "Correction equation",
    "title": "Preconditioning",
    "category": "section",
    "text": "Preconditioners can be used to improve the iterative method that solves the correction equation approximately. Although Jacobi-Davidson can be implemented with a variable or flexible preconditioner that changes each iteration, it is often more efficient to construct a fixed preconditioner for (A - tau B) or (A - tau I) for JDQZ and JDQR respectively. The motivation is that the preconditioner has to be deflated with the converged Schur vectors, which can be performed just once when the preconditioner is kept fixed.Preconditioners P are expected to implement A_ldiv_B!(P, x) which performs x = P \\ x in-place. "
},

]}
