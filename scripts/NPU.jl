using DrWatson
@quickactivate

using Flux
using VariationalFlux
using Base.Iterators
using Plots

x = 1.0:0.1:6.0
# y = x.^3

y = -x.^3 .+ 4*x.^2 .- 16


function test_npu_train1(seed,dim=3,niter=1000)
    Random.seed!(seed)
    p1 = 0.1*randn(dim)
    w = 0.1*randn(dim)
    f(x) = sum(p1.*x.^w)
    llik()=sum((y.-f.(x)).^2)

    opt = ADAM(0.001)
    data = repeated((  ), niter)
    
    ps=params(p1,p2,w)
    Flux.train!(()->llik(),ps,data,opt)
    ll =llik()
    level = 1;
    p = p1
    par=@dict seed dim niter level
    results = @dict ll p w par
    wsave(datadir("npu_train3",savename(par,"bson")),results)
end

function test_npu_vtrain1(seed,dim=3,niter=1000)
    Random.seed!(seed)
    p1 = 0.1*randn(dim)
    w = 0.1*randn(dim)
    f(x) = sum(p1.*x.^w)
    llik()=sum((y.-f.(x)).^2)

    opt = ADAM(0.001)
    data0 = repeated((  ), 1000)
    data = repeated((  ), niter)
    
    ps=params(p1,p2,w)
    Flux.train!(()->llik(),ps,data0,opt)
    vtrain_ardvb!(()->llik(),ps,data,opt;σ0=1e-2, λ0=1e-4)
    ll =llik()
    level = 1;
    p=p1;
    par=@dict seed dim niter level
    results = @dict ll p w par
    wsave(datadir("npu_vtrain3",savename(par,"bson")),results)
end


function test_npu_train2(seed,dim=3,niter=1000)
    Random.seed!(seed)
    p1 = 0.1*randn(dim)
    p2 = 0.1*randn(dim)
    w = 0.1*randn(dim)
    f(x) = sum(p1.*p2.*x.^w)
    llik()=sum((y.-f.(x)).^2)

    opt = ADAM(0.001)
    data = repeated((  ), niter)
    
    ps=params(p1,p2,w)
    Flux.train!(()->llik(),ps,data,opt)
    ll =llik()
    level = 2
    p=p1.*p2
    par=@dict seed dim niter level
    results = @dict ll p w par p1 p2
    wsave(datadir("npu_train3",savename(par,"bson")),results)
end


function test_npu_vtrain2(seed,dim=3,niter=1000)
    Random.seed!(seed)
    p1 = 0.1*randn(dim)
    p2 = 0.1*randn(dim)
    w = 0.1*randn(dim)
    f(x) = sum(p1.*p2.*x.^w)
    llik()=sum((y.-f.(x)).^2)

    opt = ADAM(0.001)
    data = repeated((  ), niter)
    data0 = repeated((  ), 1000)
    
    ps=params(p1,p2,w)
    Flux.train!(()->llik(),ps,data0,opt)
    vtrain_ardvb!(()->llik(),ps,data,opt;σ0=1e-2, λ0=1e-4)
    ll =llik()
    level = 2
    p = p1.*p2;
    par=@dict seed dim niter level
    results = @dict ll p w par
    wsave(datadir("npu_vtrain3",savename(par,"bson")),results)
end

function runtest()
    seed=rand(UInt)
    test_npu_train1(seed,4,10000)
    test_npu_train2(seed,4,10000)
    test_npu_vtrain1(seed,4,10000)
    test_npu_vtrain2(seed,4,10000)
end


using DataFrames
using DrWatson

function test_eval()
    DFt=collect_results(datadir("npu_train3"))
    DFv=collect_results(datadir("npu_vtrain3"))
    in1=map(i->DFt.par[i][:level]==1,1:200)
    in2=map(i->DFt.par[i][:level]==2,1:200)
    in1v=map(i->DFv.par[i][:level]==1,1:200)
    in2v=map(i->DFv.par[i][:level]==2,1:200)
    
    h1 = histogram(log.(DFt[in1,:ll]),nbins=100)
    h2 = histogram(log.(DFt[in2,:ll]),nbins=100)
    h1v = histogram(log.(DFv[in1,:ll]),nbins=100)
    h2v = histogram(log.(DFv[in2,:ll]),nbins=100)
    plot(h1,h2,h1v,h2v,layout=(4,1))

    h1 = histogram(map(i->minimum(abs.(DFt[i,:p])),findall(in1)),nbins=100,label="train p*x^w")
    h2 = histogram(map(i->minimum(abs.(DFt[i,:p])),findall(in2)),nbins=100,label="train p*p*x^w")
    h1v = histogram(map(i->minimum(abs.(DFv[i,:p])),findall(in1v)),nbins=100,label="ARD p*x^w")
    h2v = histogram(map(i->minimum(abs.(DFv[i,:p])),findall(in2v)),nbins=100, label="ARD p*p*x^w",xlabel="smallest parameter")
    for h in (h1,h2,h1v,h2v)
        plot(h,xlims=(0,0.5))
    end
    plot(h1,h2,h1v,h2v,layout=(4,1))

    nt1=map(i->norm(DFt[i,:p],0.1),findall(in1))
    nt2=map(i->norm(DFt[i,:p],0.1),findall(in2))
    nv1=map(i->norm(DFv[i,:p],0.1),findall(in1v))
    nv2=map(i->norm(DFv[i,:p],0.1),findall(in2v))

    scatter(log.(nt1),log.(DFt[in1,:ll]),label="train 1layer")
    scatter!(log.(nt2),log.(DFt[in2,:ll]),label="train 2layer")
    scatter!(log.(nv1),log.(DFv[in1v,:ll]),label="ARD 1layer")
    s=scatter!(log.(nv2),log.(DFv[in2v,:ll]),label="ARD 2layer")
    plot(s,xlabel="log norm(p,0.1)",ylabel="log mse")

end