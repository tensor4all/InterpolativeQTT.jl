import QuanticsGrids as QG
using InterpolativeQTT

R = 10
N = 10
a, b = -2.0, 3.0
f(x) = exp(-x^2+cos(x))
tol = 1.0e-8
P = InterpolativeQTT.getChebyshevGrid(N)

tt = InterpolativeQTT.interpolateadaptive(f, a, b, R, N; tolerance = tol)
result = InterpolativeQTT.invertqtt(tt, P; q = 1)

K_out = R - 1
S = result[K_out]

grid = QG.DiscretizedGrid{1}(R, a, b)
quanticsinds = QG.grididx_to_quantics.(Ref(grid), 1:(2^R))
ttvals = tt.(quanticsinds)

max_err = 0.0
for i in 1:(2^K_out)
    v_left = sum(P(β, 0.0) * S[i, β + 1] for β in 0:N)
    v_right = sum(P(β, 0.5) * S[i, β + 1] for β in 0:N)
    max_err = max(max_err, abs(v_left - ttvals[2i - 1]))
    max_err = max(max_err, abs(v_right - ttvals[2i]))
end

max_err < 1.0e-6
