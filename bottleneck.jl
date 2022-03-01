## Start of code

## Import dependencies
cd(@__DIR__)
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using Flux
using LinearAlgebra
using jInv.Mesh
using Printf
using Plots
using JLD
using Random
using Zygote
using DelimitedFiles
using LaTeXStrings
using Measures

include("viewers.jl")
include("gaussian-util.jl")

## Experiment parameters

d = 10 # number of dimensions

# Interaction kernel parameters
μ = 50.0                             # Mean = Value of K(x, x)
ind_sigma_value = 1.0               # sigma value
# ind_sigma_value = 0.2
σ = ind_sigma_value * ones(d)       # sigma values for interaction for each dimension

Q_value = 10                        # coefficient to change effect of obstacle

epochs = 10000         # number of iterations
ha = 0.6                # step size for a coefficient updates
hv = 0.6                # step size for velocity updates
hvtemp = 1.0


#  scales for the various costs
costPsicoeff = 10;
costAfcoeff  = 1; # costAfcoeff must be set to 1.0
costLcoeff   = 0.5;
α = [  costLcoeff;     costAfcoeff;       costPsicoeff]


# convenience functions
R = Float64
ar = x -> R.(x)

# ----------------------------------------------------------------
# Structure
# ----------------------------------------------------------------
# Build nonlocal MFG
mutable struct NonlocalMFG{R}
    a::Any # an array where each column is the list of a_i's at a particular time
    fBasis::Any # basis function
    K::AbstractArray{R} # kernel matrix
    M::AbstractArray{R} # inverse of kernel matrix
    Q::Any # function for obstacle
    Psi::Any # function corresponding to terminal function
    X0::AbstractArray{R} # training points
    w::AbstractVector{R} # quadrature weights for X0
    α::Any  # vector containing penalties for objective functions
    tspan::AbstractVector{R} # time interval
    nt::Any # number of time steps in ODE solve
    cs::Any  # contains the costs w/o penalty
end

mutable struct DataStore
    norm_grads::Float32
end

NonlocalMFG(
    a,
    fBasis,
    K::AbstractArray{R},
    M::AbstractArray{R},
    Q,
    Psi,
    X0::AbstractArray{R};
    w,
    α,
    tspan,
    nt,
) where {R<:Real} = NonlocalMFG(
    a,
    fBasis,
    K,
    M,
    Q,
    Psi,
    X0,
    w,
    α,
    tspan,
    nt,
)
function (J::NonlocalMFG{R})(vel,veltemp,hv) where {R<:Real}
    (d, _, nex) = size(J.X0)

    # step size
    h = R.((J.tspan[2] - J.tspan[1]) / J.nt)
    # final time
    tk = R.(J.tspan[1])
    # generate trajectories
    Z = getZ(vel, X0)

    cAf = zeros(R, 1, nex)
    costInteraction = 0.0
    for k = 1:J.nt
        # integrate over time to calculate costs of interaction
        at = copy(a[:, k])
        fBasisval = J.fBasis(Z[:, k, :])
        cAf += (h) .* sum(at .* fBasisval, dims = 1) # A*f cost

        costInteraction += h*0.5*sum((fBasisval*J.w).^2)
    end
    costAf = dot(vec(cAf), J.w)

    # lagrangian cost
    cL = 0.5*h*sum(sum(vel.^2, dims = 1), dims = 2)
    cL += sum(Q(Z[:, :, :]), dims = 2)
    costL = dot(vec(cL), J.w)

    # terminal cost
    phi1 = J.Psi(Z[:, end, :])
    costPsi = R.(dot(vec(phi1), vec(J.w)))

    # cost as a potential
    costPotential = costL * J.α[1] + costPsi * J.α[3] + costInteraction

    J.cs = [J.α[1]*costL, J.α[2]*costAf, J.α[3]*costPsi, costPotential, costInteraction]

    subtr = vec(vel-veltemp)
    prox = 0.5/hv*dot(subtr, subtr)

    Jc = dot(vec(J.cs[1:3]), [1,1,1]) + prox
    return Jc
end

# generates the trajectories from the starting positions given the velocities across time
function getZ(V, X0)
    (d, nt, nex) = size(V)
    h = Float64.(1/nt)
    Z = X0
    for i in 1:nt
        Z = cat(Z,  reshape(Z[:, end, :] + V[:, i, :]*h,(d,1,nex)), dims=2)
    end
    return Z
end

# generates interpolated trajectories
function getZv2(V, X0, interpolate=2)
    (d, nt, nex) = size(V)
    h = Float64.(1/nt)
    Z = X0
    hh = 1.0/interpolate
    for i in 1:nt
        for j in 1:interpolate
            Z = cat(Z,  reshape(Z[:, end, :] + V[:, i, :]*h*hh,(d,1,nex)), dims=2)
        end
    end
    return Z
end

## Plotting function

# number of dimensions to plot
d_plot = 2

mu0_plot = [0.0, 1.0]           # Mean of initial density rho0
sig0_plot = [0.04, 0.04]        # Variance of initial density
rho0_plot = Gaussian(d_plot, sig0_plot, mu0_plot)
xtarget_plot = [0.0,  -1.0]          # Mean of target




# generates the plots using the current velocities
function plotting(X0, vel, iter, time_elapsed, save_figure = false, rel_norm_grads=1, grad_a=1)
    p1 = plot()
    # plotting recipe
    default( tickfont = (4, "arial", :grey),
             guidefont  = (4, "arial", :black),
             legendfont = (4, "arial", :grey),
             titlefont =  (7, "arial", :grey),
             legend = false,
             xformatter = :plain,
             yminorgrid = true,
             dpi = 200,
             linewidth = 1.5,
             markersize = 3.0,
            )
    # plotting grid parameters
    domain = [-2.5 2.5 -2.5 2.5]
    n = [256 256]
    MM = getRegularMesh(domain, n)
    Xc = Matrix(getCellCenteredGrid(MM)')

    # Plot initial distribution
    r0 = rho0_plot(Xc)
    # generate trajectories for each agent
    Ztraj = getZv2(vel, X0, 4)
    XX = Ztraj[:,:,1:4:end]
    nEx = size(XX, 3)
    nT = size(XX, 2)
    # plot the obstacle
    p1 = viewImage2D(Q_plot(Xc), MM, aspect_ratio = :equal, c = :amp)
    plot!(p1)

    for k = 1:nEx
        plot!(
            p1,
            XX[1, :, k],
            XX[2, :, k],
            legend = false,
            linewidth = 1,
            seriestype = [:scatter,:path],
            linecolor = :lightblue,
            aspect_ratio = :equal,
            markersize = 3,
            markeralpha = 0.4,
            markercolor = :lightblue,
            markerstrokewidth = 1,
            markerstrokealpha = 1,
            markerstrokecolor = :black,
            markerstrokestyle = :dot
        )
    end
    # include information about the norm of the gradient in the title of the plot
    norm_grads = data_stored.norm_grads
    str_title = @sprintf("d: %d agents: %d sigma: %0.2f mu: %1.2f nBasis: %d\nL: %4.2e Af: %4.2e Psi: %4.2e Obs coeff: %4.2e\ntime: %2.2e rel||grads||: %0.2e aResidual: %0.2e\ncP: %2.2e cL: %2.2e cI: %2.2e cPsi: %2.2e\n\n", d, nTrain, σ[1], μ, nBasis, costLcoeff, costAfcoeff, costPsicoeff, Q_value, time_elapsed, rel_norm_grads, grad_a, (J.cs[4].data), J.cs[1].data, J.cs[5].data, J.cs[3].data)
    title!(str_title)

    p1 = plot!(size=(340,360))
    display(p1)
    if save_figure
        savefig(filename)
    end

end

#-----------------------------------------------------------------
# Training parameters
# ----------------------------------------------------------------

# number of training samples
nTrain  = 512
# number of time steps
nTSteps = 12

# number of features for the interaction kernel
r = 512
nBasis = r
# d_ = number of dimensions to apply the interaction to
d_ = 2 # 2: first two, d: all interaction
varK̂ = 1 ./ (σ[1:d_].^2)        # 1/ variance of original kernel
K̂ = Gaussian(d_, varK̂, zeros(d_))
ωs = sample(K̂, floor(Int32, r/2))
ck = fill(2/r, floor(Int32, r/2))
# basis function to map trajectories to random features
fBasis(x) = sqrt(μ)*vcat( sqrt.(ck).*cos.((ωs' * x[1:d_, :])), sqrt.(ck).*sin.((ωs' * x[1:d_, :])))

# initialize the coefficients, a
a = randn(Float64, (nBasis, nTSteps))
a0 = copy(a)   # initial value of a

K = 1.0 * Matrix(I, nBasis, nBasis)
M = K^-1

# Initial Density: One Gaussian,  mu0 = [0.0, 1.0]
# Target : xtarget = [0.0, -1.0]
function setup_initial_density_and_target()
    mu0 = zeros(d)
    mu0[2] = 1
    sig0 = 0.01 * ones(d)                # Variance of initial density
    rho0 = Gaussian(d, sig0, mu0)
    xtarget = zeros(d)
    xtarget[2] = -1
    return mu0, sig0, rho0, xtarget
end
mu0, sig0, rho0, xtarget = setup_initial_density_and_target()

# terminal function
Psi(x) = (sum((x[1:d, :] .- xtarget) .^ 2, dims = 1))

# obstacle
obstMatrix = zeros(Int(d))
obstMatrix[1] = 1;
obstMatrix[2] = -5;
obstMatrix_plot = obstMatrix[1:2]
Q(x) = Q_value*relu.(sum(x[1:d, :, :].^2 .* obstMatrix, dims = 1))
Q_plot(x) = Q_value*relu.(sum(x[1:2, :, :].^2 .* obstMatrix_plot, dims = 1))


X0 = reshape(sample(rho0, nTrain), d, 1, nTrain)        # sample from rho0 for training data
w = fill(1.0 / size(X0, 3), size(X0, 3))    # quadrature weights for training data


## Initialize the nonlocal MFG
J = NonlocalMFG(a, fBasis, K, M, Q, Psi, X0, w , α, R.([0.0; 1.0]),  nTSteps, 0)

#----------------------------------------------------------------
# Update function
# ---------------------------------------------------------------

# intial velocities
vel     = fill(0.0, d, nTSteps, nTrain) # zero initialization
veltemp = copy(vel)

# calculate initial cost and gradients
c, back = Zygote.Tracker.forward((vel)->J(vel,veltemp,hv), vel)
grads0 = back(1)[1]
grads0 = grads0.data
norm_grads0 = norm(grads0,Inf)
grads = copy(grads0)

data_stored = DataStore(0.0)
# updates all the trajectories and coefficients
function updateAll(ha, hv, hvtemp, iter, doplot=false, plot_step=100)
    global vel, norm_grads0, grads
    start_time = time()

    (d, nex) = size(J.X0)
    h = R.((J.tspan[2] - J.tspan[1]) / J.nt)

    norm_grads = norm(grads,Inf)
    rel_norm_grads = abs(norm_grads/norm_grads0)

    veltemp = copy(vel)

    # calculate the total objective cost
    c, back = Zygote.Tracker.forward((vel)->J(vel,veltemp,hv), vel)
    # calculate the gradients
    grads = back(1)[1]
    grads = grads.data

    # update the velocities
    vel    -= grads*hv

    # generate the new trajectories
    Z = getZ(vel + hvtemp*(vel-veltemp), J.X0)

    grad_a = 0
    for tval = 1:J.nt   # integrate across time
        # integrate the features
        foZ = R.(J.fBasis(Z[:, tval, :]))
        integ = sum(reshape(J.w, 1, :).* foZ , dims = 2)

        # gradient for a at time t
        grad_at  = -J.M*a[1:nBasis, tval] + integ
        # updating a
        a[1:nBasis, tval] = a[1:nBasis, tval] + ha*grad_at
        # storing away the norm
        grad_a += norm(grad_at)^2/J.nt

    end
    # total gradient of a
    grad_a = sqrt(grad_a)

    end_time = time()
    time_elapsed = end_time-start_time

    # plot every 100 iterations
    if mod(iter+1, plot_step) == 0
        if doplot == true
            plotting(X0, vel, floor(Int,iter/50), time_elapsed, true, rel_norm_grads, grad_a)
        end
    end

    # print the costs every 10 iterations
    if mod(iter+1, 10) == 0

        costL, costAf, costPsi, costPotential, costInteraction = J.cs
        @printf "epochs: %d costPotential: %0.4f costL: %0.4f costInterac: %0.4f costPsi: %0.4f norm grads: %0.4f rel grad norm: %0.4f dualnorm: %0.4e time: %0.4f\n" iter+1 (costPotential) costL costInteraction costPsi norm_grads rel_norm_grads grad_a time_elapsed
        data_stored.norm_grads = norm_grads
    end
    return rel_norm_grads

end

## ----------------------------------------------------------------
# Run the training loop
# ---------------------------------------------------------------
filename = "./data/benchmark-obstacle-N-$(nTrain)-d-$(d)-nBasis-$(nBasis)-sigma-$(ind_sigma_value)-Q-$(Q_value)-cPsi-$(costPsicoeff)-cAf-$(costAfcoeff)-cL-$(costLcoeff).png"
print("save into $(filename)\n")
for i in 1:epochs
    rel_norm_grads = updateAll(ha, hv, hvtemp, i, true)
end


## Saving results
writefilenameparam = "./data/primal-dual-bottleneck-d-$(d)-d_-$(d_)-sigma-$(ind_sigma_value)-mu-$(μ)-N-$(nTrain)-nBasis-$(nBasis)-param.txt"
writedlm(writefilenameparam, [nTrain, nTSteps, d, nBasis, ind_sigma_value, μ,  X0, vel])

## Save plots of the trajectories at every time step to make a gif

# p1 = plot()
# # plotting recipe
# default( tickfont = (4, "arial", :grey),
#          guidefont  = (4, "arial", :black),
#          legendfont = (4, "arial", :grey),
#          titlefont =  (7, "arial", :grey),
#          legend = false,
#          xformatter = :plain,
#          yminorgrid = true,
#          dpi = 350,
#          linewidth = 1.5,
#          markersize = 3.0,
#         )
# # plotting grid parameters
# domain = [-2.5 2.5 -2.5 2.5]
# n = [256 256]
# MM = getRegularMesh(domain, n)
# Xc = Matrix(getCellCenteredGrid(MM)')
#
# # generate trajectories for each agent
# Ztraj = getZv2(vel, X0, 8)
# XX = Ztraj[:,:,1:4:end]
# nEx = size(XX, 3)
# nT = size(XX, 2)
#
# for t in 1:size(XX, 2)
#     p1 = viewImage2D(Q_plot(Xc), MM, aspect_ratio = :equal, c = :amp)
#     plot!(p1)
#     for k = 1:nEx
#         plot!(
#             p1,
#             [XX[1, t, k]],
#             [XX[2, t, k]],
#             legend = false,
#             linewidth = 1,
#             seriestype = [:scatter,:path],
#             linecolor = :lightblue,
#             aspect_ratio = :equal,
#             markersize = 3,
#             markeralpha = 0.4,
#             markercolor = :lightblue,
#             markerstrokewidth = 1,
#             markerstrokealpha = 1,
#             markerstrokecolor = :black,
#             markerstrokestyle = :dot
#         )
#     end
#     p1 = plot!(size=(340,360))
#     display(p1)
#     savefig("./gif/plot$(t)")
# end

## End of code
