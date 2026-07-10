### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ 70720a2a-7c3b-11f1-0a30-d57125bcb81a
begin
	import Pkg
	Pkg.develop(path="..")
end

# ╔═╡ 52f4d5c4-e251-4a95-a7b0-1f135a118f86
begin
	using SparsityProbes
	Pkg.add("Plots")
	using Plots
	#Pkg.add("ADTypes")
	using ADTypes
end

# ╔═╡ 36bb7b99-7043-412f-bf28-a5b23cfd087b
begin
	function plot_jacobian_sparsity(f, x, detector; marker_size=2)
		J_pattern = ADTypes.jacobian_sparsity(f, x, detector)
		fig = spy(J_pattern,
				color = :black,
				  markersize = marker_size,
				title = "Jacobian Sparsity Pattern",
				legend = false,
				  grid = false,
				  framestyle=:None,
				  yflip=true,
				  yaxis=false,
				  xaxis=false,
				 )
	end
end

# ╔═╡ a024eb6a-5598-48b4-818b-0733f9e96b1c
begin
    function tridiagonal_system(x)
        y = similar(x)
        y[1] = 2x[1] - x[2]
        for i in 2:(length(x)-1)
            y[i] = -x[i-1] + 2x[i] - x[i+1]
        end
        y[end] = -x[end-1] + 2x[end]
        return y
    end
end

# ╔═╡ 3c41566f-2af6-4411-af33-a19efd1e705a
begin
x_test = rand(50)
detector = ChunkedDetector(5)

plot_jacobian_sparsity(tridiagonal_system, x_test, detector)
end

# ╔═╡ 403e03b7-1508-43ee-94cb-2068b55078d9
begin
bf_det = BloomFilterDetector(30, 10)

plot_jacobian_sparsity(tridiagonal_system, x_test, bf_det)
end

# ╔═╡ c7c2abe6-ea88-4e17-8fbf-be47b7aa367d
begin
hbf_det = HierarchicalBloomFilterDetector(30, 10)

plot_jacobian_sparsity(tridiagonal_system, x_test, hbf_det)
end

# ╔═╡ 57af1b83-f0fa-4003-ada3-84b3d06345c7
begin
	function plot_bloom_coloring(Rbar::AbstractMatrix{Bool})
        colors = SparsityProbes._color_columns(Rbar)
        num_colors = maximum(colors; init = 0)
        
        if num_colors == 0
            return plot(title="Empty Matrix (0 Colors)")
        end
    

        rows = Int[]
        cols = Int[]
        point_colors = Int[]
    
        for j in axes(Rbar, 2)
            for i in axes(Rbar, 1)
                if Rbar[i, j]
                    push!(rows, i)
                    push!(cols, j)
                    push!(point_colors, colors[j]) 
                end
            end
        end
    
        fig = scatter(cols, rows, 
            group = point_colors,
            markersize = 5,
            markerstrokewidth = 0,
            yflip = true,
            xlims = (0.5, size(Rbar, 2) + 0.5),
            ylims = (0.5, size(Rbar, 1) + 0.5),
            title = "Hierarchical Bloom Coloring ($num_colors Colors)",
            xlabel = "Inputs (Columns)",
            ylabel = "Outputs (Rows)",
            framestyle = :box,
            grid = false,
            legend = :outerright,
            legend_title = "Color ID"
        )
                
        return fig
    end
end

# ╔═╡ b1527d52-46ad-42ee-845a-79f80e09f5ee
begin
    function overlapping_system(x)
        y = similar(x)
        y[1] = x[1] * x[2]
        for i in 2:(length(x)-1)
            y[i] = x[i-1] * x[i] * x[i+1] 
        end
        y[end] = x[end-1] * x[end]
        return y
    end
    
    # parameters
    x_test0 = rand(20)
    filter_size_m = 30
    num_hashes_k = 2
    
    # run the  logic in _jacobian_sparsity_hierarchical_bloom
    S = SparsityProbes._bloomseed(length(x_test0), num_hashes_k, filter_size_m)
    Q = SparsityProbes._probe(overlapping_system, S)
    Rbar = SparsityProbes._bloomharvest(Q, S, num_hashes_k)

    # visualize Rbar
    plot_bloom_coloring(Rbar)
end

# ╔═╡ Cell order:
# ╠═70720a2a-7c3b-11f1-0a30-d57125bcb81a
# ╠═52f4d5c4-e251-4a95-a7b0-1f135a118f86
# ╠═36bb7b99-7043-412f-bf28-a5b23cfd087b
# ╠═a024eb6a-5598-48b4-818b-0733f9e96b1c
# ╠═3c41566f-2af6-4411-af33-a19efd1e705a
# ╠═403e03b7-1508-43ee-94cb-2068b55078d9
# ╠═c7c2abe6-ea88-4e17-8fbf-be47b7aa367d
# ╠═57af1b83-f0fa-4003-ada3-84b3d06345c7
# ╠═b1527d52-46ad-42ee-845a-79f80e09f5ee
