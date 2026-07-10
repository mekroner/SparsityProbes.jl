.PHONY: benchmark plot


benchmark:
	julia --project=benchmark benchmark/benchmark.jl

bench_plot:
	julia --project=benchmark benchmark/plot_benchmark_results.jl