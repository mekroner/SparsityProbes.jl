using Plots
using BenchmarkTools

# Forcing the backend into "headless" mode 
ENV["GKSwstype"] = "100"

loaded_results = BenchmarkTools.load("benchmark/results.json")[1];
println("Loaded benchmark results from results.json")

chunk_sizes = Int[]
times_ms = Float64[]
memory_mib = Float64[]

for (key, trial) in loaded_results["chunked"]
    cs = parse(Int, replace(key, "cs=" => ""))
    push!(chunk_sizes, cs)
    push!(times_ms, trial.times[1] / 1e6)  
    push!(memory_mib, trial.memory / (1024^2))  
end

println("Parsed benchmark results.")

perm = sortperm(chunk_sizes)
chunk_sizes = chunk_sizes[perm]
times_ms = times_ms[perm]
memory_mib = memory_mib[perm]

p1 = plot(
    chunk_sizes, times_ms,
    xlabel="Chunk Size",
    ylabel="Time (ms)",
    title="Runtime Time vs Chunk Size",
    legend=false,
    marker=:circle,
)

p2 = plot(chunk_sizes, memory_mib, 
    title="Memory vs Chunk Size", 
    xlabel="Chunk Size", 
    ylabel="Memory (MiB)", 
    marker=:square, 
    legend=false,
    color=:red,
)

fig = plot(p1, p2, layout=(2, 1), size=(600, 600))

savefig(fig, "benchmark/benchmark_results.png")
println("Benchmark results plotted and saved to benchmark_results.png")